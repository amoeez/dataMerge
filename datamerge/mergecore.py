#!/usr/bin/env python
# coding: utf-8
import logging
import time
from attrs import field, define, validators, Factory
import numpy as np
import pandas as pd
from .findscaling import findScaling
from .dataclasses import (
    mergeConfigObj,
    mergedDataObj,
    scatteringDataObj,
    rangeConfigObj,
)
from typing import List, Optional
from statsmodels.stats.weightstats import DescrStatsW
from multiprocessing.pool import ThreadPool as Pool
import importlib.util
import sys


@define
class mergeCore:
    """The core merging function, most of the actual action happens here"""

    mergeConfig: mergeConfigObj = field(
        validator=validators.instance_of(mergeConfigObj)
    )
    # dataList is hardly used, instead the scatteringDataObjs in ranges is used. Depreciate?
    dataList: List[scatteringDataObj] = field(validator=validators.instance_of(list))
    # the following will be constructed in here:
    ranges: list = field(default=Factory(list), validator=validators.instance_of(list))
    preMData: Optional[pd.DataFrame] = field(
        default=None,
        validator=validators.optional(validators.instance_of(pd.DataFrame)),
    )
    mData: mergedDataObj = field(
        default=Factory(mergedDataObj), validator=validators.instance_of(mergedDataObj)
    )
    poolSize: int = field(
        default=50, validator=validators.instance_of(int), converter=int
    )

    def constructRanges(self, dataList: list) -> None:
        """constructs a full range list for the complete set of loaded data"""
        assert (
            len(self.ranges) == 0
        ), "constructRanges should only be done starting from an empty range list"
        for ri, dataset in enumerate(dataList):
            self.ranges += [
                rangeConfigObj(
                    rangeId=ri,
                    scatteringData=dataset,
                )
            ]
        return

    def sortRanges(self, reverse: bool = False) -> None:
        """Sorts self.ranges by qMin from smallest to largest (ascending=True)"""
        self.ranges.sort(key=lambda x: x.scatteringData.qMin(), reverse=reverse)
        # reset rangeId
        # rid = 0
        # for drange in self.ranges:
        #     drange.rangeId=rid
        #     rid += 1
        [setattr(drange, "rangeId", rid) for rid, drange in enumerate(self.ranges)]
        return

    def updateRanges(self, rangesConfigList: list) -> None:
        """
        Gets the ranges list from the configuration yaml, and applies the requested
        range settings to the rangelist in self.ranges.
        Range configuration updates can be specified by rangeId or by a findByConfig number.
        """
        if len(rangesConfigList) == 0:
            return  # nothing to do
        for rangeConfig in rangesConfigList:
            # find out if specified by rangeId or findByConfig
            if rangeConfig.rangeId == -1:  # find by configuration number
                assert (
                    rangeConfig.findByConfig is not None
                ), "if rangeId is set to -1, a findByConfig number must be specified"
                resultList = [
                    drange
                    for drange in self.ranges
                    if drange.scatteringData.configuration == rangeConfig.findByConfig
                ]
                # assert len(resultList)>=1, 'findByConfig should result in at least one unique datafile. Did not find any matches'
                if len(resultList) == 0:
                    logging.debug(
                        f"Did not find a matching range for {rangeConfig.findByConfig=}"
                    )
                    continue  # Nothing to do, no matching ranges found
            elif rangeConfig.rangeId >= 0:  # defined by rangeId
                if rangeConfig.rangeId <= len(self.ranges):
                    resultList = [self.ranges[rangeConfig.rangeId]]  # one-element list
                else:
                    logging.warning(
                        f"Range configuration change requested for rangeId {rangeConfig.rangeId}, but this ID goes beyond the loaded files. skipping"
                    )
                    continue
            else:
                logging.warning(
                    f"{rangeConfig.rangeId=} is not a valid number, must be either -1 or the index of a range"
                )
                continue
            for drange in resultList:
                # update settings for all matching configurations
                [
                    setattr(drange, key, value)
                    for key, value in rangeConfig.items()
                    if (key in dir(drange))
                    and not (key in ["rangeId", "scatteringData"])
                ]
        return

    def rangeObjAsDataframe(self, rangeObj: rangeConfigObj) -> pd.DataFrame:
        """
        Returns a dataframe from the rangeobj, with all the corrections and adjustments applied:
          - scatteringData.mask
          - qMin, qMax
          - scale factor (if applicable)
          - eMin
          - qeMin
        """
        assert rangeObj.scatteringData is not None, logging.error(
            "ScatteringData cannot be none when exporting range object as DataFrame"
        )
        df = rangeObj.scatteringData.asPandas(
            maskArray=rangeObj.scatteringData.Mask
            | rangeObj.scatteringData.returnMaskByQRange(
                qMin=rangeObj.qMinPreset, qMax=rangeObj.qMaxPreset
            ),
            scaling=rangeObj.scale,
        )
        df.ISigma.clip(lower=self.mergeConfig.eMin * df.I, inplace=True)
        df.QSigma.clip(lower=self.mergeConfig.qeMin * df.Q, inplace=True)
        return df

    def autoScale(self) -> None:
        # if any of the autoscaling dials is set to something else than itself, find the scaling factor between those datasets
        for drange in self.ranges:  # dfn = drange.rangeId, idf = drange
            drange.scale = 1.0  # reset in case of change of heart
            # if self.rangesDf.loc[dfn, 'autoscaletorange'] != dfn:
            if drange.autoscaleToRange is not None:
                logging.debug(
                    f"autoscaling range: {drange.rangeId} to index {drange.autoscaleToRange}"
                )
                # find scaling factor
                assert drange.autoscaleToRange <= len(
                    self.ranges
                ), f"{drange.autoscaleToRange=} must refer to an index within the number of ranges available {len(self.ranges)}"
                oRange = self.ranges[drange.autoscaleToRange]
                fs = findScaling(
                    self.rangeObjAsDataframe(
                        oRange
                    ),  # original data, asPandas() returns copy
                    self.rangeObjAsDataframe(
                        drange
                    ),  # data to scale, asPandas() returns a copy
                    backgroundFit=False,  # just the scaling factor, please, no additional background.
                )
                drange.scale = float(fs.sc[0])
                logging.info(
                    f"Scaling added to rangeindex: {drange.rangeId}: {float(fs.sc[0])}"
                )
        return

    def concatAllUnmergedData(self) -> None:
        """
        Reads all the datasets in self.dataList, applying masks, limits and scaling factors where necessary.
        All the datasets are then put in a single scatteringDataObj instance in self.preMData,
        ready to be merged.
        """
        # nothing to do
        if len(self.ranges) == 1:
            df = self.rangeObjAsDataframe(self.ranges[0])
        else:
            with Pool(self.poolSize) as pool:
                dfs = [
                    d
                    for d in pool.imap_unordered(self.rangeObjAsDataframe, self.ranges)
                ]
            df = pd.concat(dfs, ignore_index=True, sort=False)
        self.preMData = df.dropna(thresh=2)
        return

    def sortUnmergedData(self) -> None:
        if self.preMData is not None:
            self.preMData.sort_values(by="Q", inplace=True)
        return

    def nonZeroQMin(self) -> float:
        """Returns the smallest nonzero q value for all input ranges for starting the binning at."""
        qMin = np.inf
        for drange in self.ranges:
            if drange.qMinPreset is not None:
                qMin = np.min([qMin, drange.qMinPreset])
            elif drange.scatteringData.qMin() > 0:
                qMin = np.min([qMin, drange.scatteringData.qMin()])
        return qMin

    def nonZeroQMax(self) -> float:
        """Returns the largest non-infinie q value for all input datasets, for ending the binning at."""
        qMax = 0.0
        for drange in self.ranges:
            if drange.qMaxPreset is not None:
                qMax = np.max([qMax, drange.qMaxPreset])
            elif drange.scatteringData.qMax() < np.inf:
                qMax = np.max([qMax, drange.scatteringData.qMax()])
        return qMax

    def createBinEdges(self) -> np.ndarray:
        binEdges = list()
        assert self.mergeConfig.outputRanges is not None, logging.error(
            "at least one output range should be specified"
        )
        for rangeId, outRange in enumerate(self.mergeConfig.outputRanges):
            qEnd = self.nonZeroQMax()
            qStart = (
                outRange.qCrossover
                if (outRange.qCrossover != 0) and (np.isfinite(outRange.qCrossover))
                else self.nonZeroQMin()
            )
            if rangeId < (
                len(self.mergeConfig.outputRanges) - 1
            ):  # set qEnd to the start of the next range
                newQEnd = self.mergeConfig.outputRanges[rangeId + 1].qCrossover
                if np.isfinite(newQEnd):
                    qEnd = newQEnd
            if outRange.QScaling == "log":
                binEdges += [np.geomspace(qStart, qEnd, num=outRange.nbins + 1)]
            else:
                binEdges += [np.linspace(qStart, qEnd, num=outRange.nbins + 1)]

        be = np.concatenate(binEdges, dtype=float)
        # add a little to the end to ensure the last datapoint is included:
        assert len(be > 3), logging.error(
            "number of bins in total range must exceed at least 3.."
        )
        be[-1] = be[-1] + 1e-3 * (be[-1] - be[-2])
        return be

    def mergyMagic(self, binEdges: np.ndarray, calcSEMw: bool = False) -> None:
        # define weighted standard error on the mean:
        def SEMw(x, w):
            """
            function adapted from: https://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
            citing: The main reference is this paper, by Donald F. Gatz and Luther Smith, where 3 formula based estimators are compared with bootstrap results. The best approximation to the bootstrap result comes from Cochran (1977):
            side note: this provides rubbish results.
            """
            n = len(w)
            # dummy = [print(f"x: {xi}, {wi}") for xi, wi in zip(x, w)]
            xWbar = np.average(x, weights=w)
            wbar = np.mean(w)
            out = (
                n
                / ((n - 1) * np.sum(w) ** 2)
                * (
                    np.sum((w * x - wbar * xWbar) ** 2)
                    - 2 * xWbar * np.sum((w - wbar) * (w * x - wbar * xWbar))
                    + xWbar**2 * np.sum((w - wbar) ** 2)
                )
            )
            return out

        self.mData = mergedDataObj(
            Q=np.full(len(binEdges) - 1, np.nan),
            I=np.full(len(binEdges) - 1, np.nan),
            IStd=np.full(len(binEdges) - 1, np.nan),
            ISEM=np.full(len(binEdges) - 1, np.nan),
            ISEMw=np.full(len(binEdges) - 1, np.nan),
            IEPropagated=np.full(len(binEdges) - 1, np.nan),
            ISigma=np.full(len(binEdges) - 1, np.nan),
            QStd=np.full(len(binEdges) - 1, np.nan),
            QSEM=np.full(len(binEdges) - 1, np.nan),
            QSigma=np.full(len(binEdges) - 1, np.nan),
            Mask=np.full(
                len(binEdges) - 1, True, dtype=bool
            ),  # all masked until filled in
            Singles=np.full(
                len(binEdges) - 1, False, dtype=bool
            ),  # Set to True for bins containing just one datapoint
        )
        logging.debug(f"{binEdges=}")
        assert self.preMData is not None, logging.warning(
            "self.preMData cannot be none at the merging step"
        )

        edgeIndices = np.searchsorted(
            self.preMData.Q.values, binEdges
        )  # last edge should be slightly outside last
        # we can precalculate the weights for all datapoints:
        if self.mergeConfig.IEWeighting:
            self.preMData["wt"] = np.abs(
                self.preMData.I / (self.preMData.ISigma**2)
            )  # inverse relative weight per point if desired.
        else:
            self.preMData["wt"] = np.abs(
                self.preMData.I * 0.0 + 1
            )  # no datapoint weighting

        def binDfRangeByIndex(binN: int) -> None:
            lowerIndex, upperIndex = edgeIndices[binN], edgeIndices[binN + 1]
            rangeLen = upperIndex - lowerIndex
            if rangeLen == 0:  # nothing in bin
                return
            dfRange = self.preMData.iloc[lowerIndex:upperIndex, :]
            if rangeLen == 1:  # one datapoint in bin
                # might not be necessary to do this..
                # can't do stats on this:
                self.mData.Q[binN] = float(dfRange.Q)
                self.mData.I[binN] = float(dfRange.I)
                self.mData.IStd[binN] = float(dfRange.ISigma)
                self.mData.ISEM[binN] = float(dfRange.ISigma)
                self.mData.ISEMw[binN] = float(dfRange.ISigma)
                self.mData.IEPropagated[binN] = float(dfRange.ISigma)
                self.mData.ISigma[binN] = float(dfRange.ISigma)
                self.mData.QStd[binN] = float(dfRange.QSigma)
                self.mData.QSEM[binN] = float(dfRange.QSigma)
                self.mData.QSigma[binN] = float(dfRange.QSigma)
                self.mData.Singles[binN] = True
                self.mData.Mask[binN] = False
                return

            else:  # multiple datapoints in bin
                name = 'binstats'
                # check if cython modules are available
                if (spec := importlib.util.find_spec(name)) is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    logging.info("Faster cythonized rebinnning statistics functions will be used")
                    I_values = dfRange.I.values
                    Q_values = dfRange.Q.values
                    I_sigma = dfRange.ISigma.values
                    weights = dfRange.wt.values

                    if self.preMData.dtypes.I == 'float64':
                        self.mData.I[binN] = module.weighted_mean(I_values, weights)
                        self.mData.IStd[binN] = module.std_ddof(I_values, weights)
                        # following suggestion regarding V1/V2 from: https://groups.google.com/forum/#!topic/medstats/H4SFKPBDAAM
                        self.mData.ISEM[binN] = module.sem(I_values, weights)
                        
                        if calcSEMw:
                            self.mData.ISEMw[binN] = module.sem_weighted(I_values, weights)
                    else:
                        self.mData.I[binN] = module.weighted_mean_sp(I_values, weights)
                        self.mData.IStd[binN] = module.std_ddof_sp(I_values, weights)
                        self.mData.ISEM[binN] = module.sem_sp(I_values, weights)
                        
                        if calcSEMw:
                            self.mData.ISEMw[binN] = module.sem_weighted_sp(I_values, weights)

                    if self.preMData.dtypes.ISigma == 'float64':
                        self.mData.ISigma[binN] = module.sigma(I_sigma, weights)
                    else: 
                        self.mData.ISigma[binN] = module.sigma_sp(I_sigma, weights)
                    self.mData.IEPropagated[binN] = module.propagated_error(self.mData.ISEM[binN], self.mData.ISigma[binN], self.mData.I[binN], self.mergeConfig.eMin)

                    self.mData.Q[binN] = module.weighted_mean(Q_values, weights)
                    self.mData.QStd[binN] = module.std_ddof(Q_values, weights)
                    self.mData.QSEM[binN] = module.sem(Q_values, weights)
                    self.mData.QSigma[binN] = module.propagated_error(self.mData.QSEM[binN], -0.1, self.mData.Q[binN], self.mergeConfig.eMin) # instead of second value which was I sigma in the above call of the func, set to negative
                    self.mData.Mask[binN] = False
                else:
                    logging.info("Slower python rebinnning statistics functions will be used")
                    # exploit the DescrStatsW package from statsmodels
                    DSI = DescrStatsW(dfRange.I, weights=dfRange.wt)
                    DSQ = DescrStatsW(dfRange.Q, weights=dfRange.wt)
                    self.mData.Q[binN] = DSQ.mean
                    self.mData.I[binN] = DSI.mean
                    self.mData.ISigma[binN] = (
                        np.sqrt(((dfRange.wt * dfRange.ISigma) ** 2).sum())
                        / dfRange.wt.sum()
                    )
                    self.mData.IStd[binN] = DSI.std
                    # following suggestion regarding V1/V2 from: https://groups.google.com/forum/#!topic/medstats/H4SFKPBDAAM
                    self.mData.ISEM[binN] = DSI.std * np.sqrt(
                        (dfRange.wt**2).sum() / (dfRange.wt.sum()) ** 2
                    )
                    if calcSEMw:
                        self.mData.ISEMw[binN] = SEMw(
                            dfRange.I, dfRange.wt
                        )  # adds considerable time, and we're not using it at the mo.
                    self.mData.IEPropagated[binN] = np.max(
                        [
                            self.mData.ISEM[binN],
                            self.mData.ISigma[binN],
                            DSI.mean * self.mergeConfig.eMin,
                        ]
                    )
                    self.mData.QStd[binN] = DSQ.std
                    self.mData.QSEM[binN] = DSQ.std * np.sqrt(
                        (dfRange.wt**2).sum() / (dfRange.wt.sum()) ** 2
                    )
                    self.mData.QSigma[binN] = np.max(
                        [self.mData.QSEM[binN], DSQ.mean * self.mergeConfig.qeMin]
                    )
                    self.mData.Mask[binN] = False
            return

        # separate:
        with Pool(self.poolSize) as pool:
            _ = [
                i
                for i in pool.imap_unordered(
                    binDfRangeByIndex, range(len(binEdges) - 1)
                )
            ]

        return

    def returnMaskedOutput(
        self, maskMasked: bool = True, maskSingles: bool = False
    ) -> mergedDataObj:
        newMDO = mergedDataObj()
        assert self.mData.Mask is not None, logging.warning(
            "Mask array must be filled at this point"
        )
        mask = np.zeros(shape=self.mData.Mask.shape, dtype=bool)
        if maskMasked:
            mask += self.mData.Mask
        if maskSingles:
            assert self.mData.Singles is not None, logging.warning(
                "Singles array must be filled at this point"
            )
            mask += self.mData.Singles
        keyList = ["Q"]  # Q must be set first
        keyList += [key for key in newMDO.keys() if not key in keyList]

        assert mask.sum() < len(mask), logging.error(
            "Cannot output anything, all values masked"
        )
        for key in keyList:
            setattr(newMDO, key, self.mData[key][~mask])
        return newMDO

    def run(self) -> mergedDataObj:
        # config is already read, and the raw data has been loaded into a list of scatteringData objects
        # construct initial list of ranges
        starttime = time.time()
        logging.info("1. constructing ranges. t=0")
        self.constructRanges(self.dataList)
        # sort by qMin so that index 0 is the one with the smallest qMin
        logging.info(f"2. sorting ranges. t={time.time() - starttime}")
        self.sortRanges()
        [
            logging.debug(
                f"rID: {dr.rangeId}, has SDL: {dr.scatteringData is not None}, qMin: {dr.scatteringData.qMin()}"
            )
            for dr in self.ranges
        ]
        # update ranges with custom configuration if necessary
        if self.mergeConfig.ranges is not None:
            logging.info(f"2.1 updating ranges. t={time.time() - starttime}")
            logging.debug(f"{self.mergeConfig.ranges=}")
            self.updateRanges(self.mergeConfig.ranges)
        # determine scaling factors
        logging.info(f"3. applying autoscaling, t={time.time() - starttime}")
        # self.autoScale()
        # just checking it makes it to here.
        o = [
            f"{dr.rangeId}({dr.scatteringData.configuration}): {dr.scale}"
            for dr in self.ranges
        ]
        logging.info(f"AutoScale done, scaling factors: {o}")
        logging.debug(f"{self.mergeConfig.outputRanges=}")
        # do I need to resort the data by Q? I don't think so... was part of the original though.
        # read all the data into a single dataframe, taking care of scaling, clipping and masking
        logging.info(f"4. concatenating original data, t={time.time() - starttime}")
        self.concatAllUnmergedData()
        # Sort for posterity
        logging.info(f"5. Sorting unmerged data, t={time.time() - starttime}")
        self.sortUnmergedData()
        # apply mergyMagic to the list of q Edges.
        logging.info(f"6. creating bin edges, t={time.time() - starttime}")
        binEdges = self.createBinEdges()
        logging.info(f"6. merging within bin edges, t={time.time() - starttime}")
        # this is the bottleneck... not surprising but still.
        self.mergyMagic(binEdges=binEdges)
        # filter result
        logging.info(
            f"7. filtering out invalid points from merged data with {self.mergeConfig.maskMasked=} and {self.mergeConfig.maskSingles=}, t={time.time() - starttime}"
        )
        filteredMDO = self.returnMaskedOutput(
            maskMasked=self.mergeConfig.maskMasked,
            maskSingles=self.mergeConfig.maskSingles,
        )
        logging.info(f"7.1 done filtering, t={time.time() - starttime}")

        return filteredMDO
