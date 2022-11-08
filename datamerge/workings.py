# Code cannibalized, ui elements removed to automate... 
# Code leftovers of previous iteration, just for cannibalization. None of this should be used directly. 
import pandas as pd


class dataMerge(object):
    def readDfList(self, rangesDf, eMin=0.01):
        """
        Reads a dataframe version of the above, takes q limits into account
        """
        item = rangesDf.iloc[0]
        df = pd.DataFrame(columns=["Q", "I", "IError"])
        for itemnum, item in rangesDf.iterrows():
            print("working on filename: {}".format(item.filename))
            df2 = self.read_nexus(
                item.filepath, qmin=item.qmin, qmax=item.qmax, eMin=eMin
            )
            # apply scale:
            df2.I *= item.scale
            df2.IError *= item.scale
            df = df.append(df2, ignore_index=True, sort=False)

        return df

    def mergyMagic(self, df, nBin=500, qBounds=None):

        if qBounds is None:
            qMin, qMax = df.Q.min(), df.Q.max()
        else:
            qMin, qMax = np.min(qBounds), np.max(qBounds)

        # define weighted standard error on the mean:
        def SEMw(x, w):
            # function adapted from: https://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
            # citing: The main reference is this paper, by Donald F. Gatz and Luther Smith, where 3 formula based estimators are compared with bootstrap results. The best approximation to the bootstrap result comes from Cochran (1977):
            # side note: this provides rubbish results.

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

        # pass through everything outside the bin range

        # prepare bin edges:
        binEdges = np.logspace(np.log10(qMin), np.log10(qMax), num=nBin + 1)
        binDat = pd.DataFrame(
            data={
                "Q": np.full(nBin, np.nan),  # mean Q
                "I": np.full(nBin, np.nan),  # mean intensity
                "IStd": np.full(
                    nBin, np.nan
                ),  # standard deviation of the mean intensity
                "ISEM": np.full(
                    nBin, np.nan
                ),  # standard error on mean of the mean intensity (maybe, but weighted is hard.)
                "ISEMw": np.full(
                    nBin, np.nan
                ),  # weighted standard error on mean of the mean intensity
                "IError": np.full(nBin, np.nan),  # Propagated errors of the intensity
                "IE": np.full(nBin, np.nan),  # Combined error estimate of the intensity
                "QStd": np.full(nBin, np.nan),  # standard deviation of the mean Q
                "QSEM": np.full(nBin, np.nan),  # standard deviation of the mean Q
                # "QError": np.full(nBin, np.nan), # Propagated errors on the mean Q
                "QE": np.full(nBin, np.nan),  # Combined errors on the mean Q
            }
        )

        # add a little to the end to ensure the last datapoint is captured:
        binEdges[-1] = binEdges[-1] + 1e-3 * (binEdges[-1] - binEdges[-2])

        # now do the binning per bin.
        for binN in range(len(binEdges) - 1):
            # this almost works, except for the very last datapoint.
            # Perhaps shift the last edge a little outward?
            dfRange = df.query(
                "{} <= Q < {}".format(binEdges[binN], binEdges[binN + 1])
            ).copy()
            if len(dfRange) == 0:
                # print("ping")
                pass
            elif len(dfRange) == 1:
                # might not be necessary to do this..
                # print("pong")
                # can't do stats on this:
                binDat.Q.loc[binN] = float(dfRange.Q)
                binDat.I.loc[binN] = float(dfRange.I)
                binDat.IError.loc[binN] = float(dfRange.IError)
                binDat.IStd.loc[binN] = float(dfRange.IError)
                binDat.ISEM.loc[binN] = float(dfRange.IError)
                binDat.ISEMw.loc[binN] = float(dfRange.IError)
                binDat.IE.loc[binN] = float(dfRange.IError)
                # binDat.QStd.loc[binN]   = float(dfRange.QError)
                # binDat.QSEM.loc[binN]   = float(dfRange.QError)
                # binDat.QE.loc[binN]     = binDat.QSEM.loc[binN]

            else:
                # dfRange.IError.clip_lower(dfRange.I * 0.01) # clip to minimum uncertainty
                # dfRange["wt"] = np.abs((dfRange.I / dfRange.IError)**2) # inverse relative weight per point
                dfRange["wt"] = np.abs(
                    dfRange.I / (dfRange.IError**2)
                )  # inverse relative weight per point
                # dfRange.wt /= dfRange.wt.max() # normalization, probably not necessary
                DSI = DescrStatsW(dfRange.I, weights=dfRange.wt)
                DSQ = DescrStatsW(dfRange.Q, weights=dfRange.wt)

                binDat.Q.loc[binN] = DSQ.mean
                binDat.I.loc[binN] = DSI.mean
                binDat.IError.loc[binN] = (
                    np.sqrt(((dfRange.wt * dfRange.IError) ** 2).sum())
                    / dfRange.wt.sum()
                )
                binDat.IStd.loc[binN] = DSI.std
                # following suggestion regarding V1/V2 from: https://groups.google.com/forum/#!topic/medstats/H4SFKPBDAAM
                binDat.ISEM.loc[binN] = DSI.std * np.sqrt(
                    (dfRange.wt**2).sum() / (dfRange.wt.sum()) ** 2
                )
                binDat.ISEMw.loc[binN] = SEMw(dfRange.I, dfRange.wt)
                binDat.IE.loc[binN] = np.max(
                    [binDat.ISEM[binN], binDat.IError[binN], DSI.mean * 1e-2]
                )
                binDat.QStd.loc[binN] = DSQ.std
                binDat.QSEM.loc[binN] = DSQ.std * np.sqrt(
                    (dfRange.wt**2).sum() / (dfRange.wt.sum()) ** 2
                )
                binDat.QE.loc[binN] = binDat.QSEM.loc[binN]

        # remove empty bins
        binDat.dropna(thresh=4, inplace=True)
        return binDat

    def storeHDF5(self):
        # use this file as a base:
        filename = Path(self.workon, self.outputfilename)
        # remove if already exists:
        if filename.is_file():
            filename.unlink()
        # this is the way (to make nexusformat write the base structure)
        nxf = nx.NXroot()
        nxf = nxf.save(filename)
        # now we can start:
        nxf[f"/datamerge/"] = nx.NXentry()
        nxf[f"/datamerge/settings"] = nx.NXgroup()

        # store information on the input datasets and settings:
        for dfn, item in self.rangesDf.iterrows():
            # link original dataset in the structure
            nxf[f"/entry{dfn}"] = nx.NXlink("/processed", file=item.filepath)
            nxf[f"/datamerge/settings/{dfn}"] = nx.NXgroup()
            # store settings from rangesDf associated with each dataset
            for key, val in item.items():
                if isinstance(val, Path):
                    val = val.as_posix()
                nxf[f"/datamerge/settings/{dfn}"].attrs[key] = val

        # now store the resulting binned dataset itself
        nxf[f"/datamerge/result"] = nx.NXdata(
            nx.NXfield(self.binDat.I, name="I"),
            axes=(nx.NXfield(self.binDat.Q, name="Q")),
            errors=nx.NXfield(self.binDat.IE, name="IE"),
        )  # IE is the combined uncertainty estimate

        # store the sample name and sample owner
        nxf[f"/sample_name"] = self.sampleName
        nxf[f"/sample_owner"] = self.sampleOwner
        # store the remaining uncertainty estimators alongside the dataset:
        binDatAxes = list(self.binDat.keys()).copy()
        [binDatAxes.remove(key) for key in ["Q", "I", "IE"]]
        for key in binDatAxes:
            nxf[f"/datamerge/result/{key}"] = nx.NXfield(
                dmui.binDat.loc[:, key], name=key
            )
        # link the Q uncertainties:
        nxf[f"/datamerge/result/Q"].attrs["uncertainties"] = "QE"
        # also set as resolution for now
        nxf[f"/datamerge/result/Q"].attrs["resolutions"] = "QE"
        # set the default path to follow
        nxf[f"/datamerge"].attrs["default"] = "result"
        nxf.attrs["default"] = "datamerge"
        # link main SASentry to datamerge dataset
        nxf["/entry"] = nx.NXlink("/datamerge")
        # canSAS compatibiity
        nxf["/datamerge"].attrs["canSAS_class"] = "SASentry"
        nxf["/datamerge"].attrs["version"] = "1.0"
        nxf["/datamerge/definition"] = nx.NXfield("NXcanSAS")
        nxf["/datamerge/run"] = nx.NXfield(0)
        nxf["/datamerge/title"] = nx.NXfield(
            f"merged dataset from {len(self.rangesDf)} datasets"
        )
        nxf["/datamerge/result"].attrs["canSAS_class"] = "SASdata"
        nxf["/datamerge/result"].attrs["I_axes"] = "Q"
        nxf["/datamerge/result"].attrs["Q_indices"] = 0
        nxf["/datamerge/result/Q"].attrs["units"] = "1/nm"
        nxf["/datamerge/result/I"].attrs["units"] = "1/m"
        # why doesn't this path exist?
        # nxf['/datamerge/result/IE'].attrs['units'] = "1/m"

    def binnit(self):
        # reloading data with limits
        self.df = self.readDfList(self.rangesDf)
        # and remerging:
        self.df.sort_values("Q", inplace=True)
        # small-angle range:
        binDat_range1 = self.mergyMagic(
            self.df, nBin=self.nbinsLower, qBounds=[self.df.Q.min(), self.qCrossover]
        )
        # wide-angle, XRD range:
        binDat_range2 = self.mergyMagic(
            self.df, nBin=self.nbinsUpper, qBounds=[self.qCrossover, self.df.Q.max()]
        )
        # add the two ranges together:
        self.binDat = pd.concat([binDat_range1, binDat_range2], ignore_index=True)
        # export to CSV:
        self.binDat.to_csv(
            Path(self.workon, f"{self.outputfilename.stem}_tunedMerge.dat"),
            columns=["Q", "I", "IE"],
            sep="\t",
            index=False,
            header=None,
        )

    def plotFigure(self):
        fh, (ah) = plt.subplots(1, figsize=[10, 8])
        # plt.sca(self.ah)
        # ah = plt.axes()
        # combined uncertainty estimate:
        ah = self.binDat.plot(
            "Q",
            "I",
            yerr="IE",
            xerr="QE",
            logx=True,
            logy=True,
            zorder=1,
            color="red",
            ax=ah,
            lw=0.5,
            label="re-binned datapoints",
            fmt="x-",
            alpha=1,
        )
        plt.plot(
            self.df.Q,
            self.df.I,
            "g+",
            color="grey",
            zorder=0,
            label=" original datapoints",
        )
        plt.plot(
            self.binDat.Q,
            self.binDat.IError,
            "g.",
            zorder=0,
            label="Propagated error on binned datapoints",
        )
        plt.plot(
            self.binDat.Q,
            self.binDat.ISEM,
            "b.",
            zorder=0,
            label="Standard error on mean on binned datapoints",
        )
        plt.plot(
            self.binDat.Q,
            self.binDat.I * 1e-2,
            "k.",
            zorder=0,
            label="1% of intensity of binned datapoints",
        )
        plt.legend()
        plt.xlabel("q (nm$^{-1}$)")
        plt.ylabel("Scattering Cross-section (m$^{-1}$)")
        plt.grid("on")

        # add ranges:
        for drn, idf in self.rangesDf.iterrows():
            ystart = 1e-4 * 4 ** (drn + 1)
            ah.hlines(
                y=ystart,
                xmin=idf.qmindetected,
                xmax=idf.qmaxdetected,
                linewidth=2,
                color="r",
            )
            ah.hlines(y=ystart, xmin=idf.qmin, xmax=idf.qmax, linewidth=2, color="b")
            print(f' * scaling reporting: {self.rangesDf.loc[drn,"scale"]}')
            ah.text(
                x=idf.qmin * 1.1,
                y=ystart * 1.4,
                s=f"Range {drn} x {self.rangesDf.loc[drn,'scale']:0.04f}",
            )

        display(fh)
        plt.savefig(Path(self.workon, f"{self.outputfilename.stem}_tunedMerge.pdf"))
        plt.savefig(
            Path(self.workon, f"{self.outputfilename.stem}_tunedMerge.png"),
            transparent=True,
        )

        fh = plt.figure(figsize=[10, 8])
        ah = plt.axes()
        # combined uncertainty estimate:
        ah = self.binDat.plot(
            "Q",
            "I",
            yerr="IE",
            xerr="QE",
            logx=True,
            logy=True,
            zorder=1,
            color="red",
            ax=ah,
            lw=0.5,
            label="re-binned datapoints",
            fmt="-",
            alpha=1,
        )
        plt.xlabel("q (nm$^{-1}$)")
        plt.ylabel("Scattering Cross-section (m$^{-1}$)")
        plt.legend(loc=0)
        plt.grid(b=True, which="major", color="black", linestyle="-", alpha=1)
        plt.minorticks_on()
        plt.grid(b=True, which="minor", color="black", linestyle=":", alpha=0.8)
        display(fh)
        plt.savefig(Path(self.workon, self.outputfilename.with_suffix(".pdf")))
        plt.savefig(
            Path(self.workon, self.outputfilename.with_suffix(".png")), transparent=True
        )

    def run(self, doPlot=True):
        # this is where the magic happens:
        # find scaling parameters:
        self.autoScale()
        # run the rebinning routine, store the resulting dat files
        self.binnit()
        # store the result in an HDF5 structure
        self.storeHDF5()
        # plot the result and save the plots
        if doPlot:
            self.plotFigure()
