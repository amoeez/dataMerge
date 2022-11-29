#!/usr/bin/env python
# coding: utf-8
# requires at least attrs version == 21.4
from pathlib import Path
import argparse
from sys import platform
import logging
from typing import Optional

# from dmdataclasses import mergedDataObj
from readersandwriters import scatteringDataObjFromNX
from readersandwriters import mergeConfigObjFromYaml
from readersandwriters import outputToNX
from mergecore import mergeCore
import sys

import matplotlib.pyplot as plt


def isMac() -> bool:
    return platform == "darwin"


def getFiles(argDict: dict) -> list:
    """
    Takes the parsed command-line argument dictionary
    and returns the list of scatteringDataObjects read from the individual files
    """
    fnames = argDict["dataFiles"]

    if len(fnames) == 1:
        if fnames[0].is_dir():
            # glob the files from the globkey in Path
            fnames = sorted(fnames[0].glob(argDict["globKey"]))
            logging.info(f"Found the following files to merge: {fnames}")
    assert len(fnames) > 0, "length of filename list to merge is zero, cannot merge."

    scatteringDataList = []
    for fname in fnames:
        assert (
            fname.is_file()
        ), f"filename {fname} does not exist. Please supply valid filenames"
        scatteringDataList += [scatteringDataObjFromNX(fname)]
    return scatteringDataList


def plotFigure(mergecore: mergeCore, ofname: Optional[Path] = None) -> None:
    fh, (ah) = plt.subplots(1, figsize=[10, 8])
    # plt.sca(self.ah)
    # ah = plt.axes()
    # combined uncertainty estimate:
    _ = plt.errorbar(
        x=mergecore.mData.Q,
        y=mergecore.mData.I,
        yerr=mergecore.mData.ISigma,
        xerr=mergecore.mData.QSigma,
        zorder=1,
        color="red",
        # ax=ah,
        lw=0.5,
        label="re-binned datapoints",
        fmt=".-",
        alpha=1,
    )
    # plt.sca(ah)
    plt.xscale("log")
    plt.yscale("log")

    # plt.plot(
    #     self.df.Q,
    #     self.df.I,
    #     "g+",
    #     color="grey",
    #     zorder=0,
    #     label=" original datapoints",
    # )
    plt.plot(
        mergecore.mData.Q,
        mergecore.mData.IEPropagated,
        "g.",
        zorder=0,
        label="Propagated error on binned datapoints",
    )
    plt.plot(
        mergecore.mData.Q,
        mergecore.mData.ISEM,
        "b.",
        zorder=0,
        label="Standard error on mean on binned datapoints",
    )
    plt.plot(
        mergecore.mData.Q,
        0.01 * mergecore.mData.I,
        "k.",
        zorder=0,
        label="1% of intensity of binned datapoints",
    )
    plt.legend()
    plt.xlabel("q (nm$^{-1}$)")
    plt.ylabel("Scattering Cross-section (m$^{-1}$)")
    plt.grid("on")

    # add ranges:
    for drange in mergecore.ranges:
        qMinActual = (
            drange.qMinPreset
            if drange.qMinPreset is not None
            else drange.scatteringData.qMin()
        )
        qMaxActual = (
            drange.qMaxPreset
            if drange.qMaxPreset is not None
            else drange.scatteringData.qMax()
        )

        ystart = 1e-4 * 4 ** (drange.rangeId + 1)
        ah.hlines(
            y=ystart,
            xmin=drange.scatteringData.qMin(),
            xmax=drange.scatteringData.qMax(),
            linewidth=4,
            color="r",
        )
        logging.info(f"{qMinActual=}, {qMaxActual=}")
        ah.hlines(
            y=ystart,
            xmin=qMinActual,
            xmax=qMaxActual,
            linewidth=2,
            color="b",
        )
        print(f" * scaling reporting: {drange.scale:0.04f}")
        ah.text(
            x=drange.scatteringData.qMin() * 1.1,
            y=ystart * 1.4,
            s=f"Range {drange.rangeId} x {drange.scale:0.04f}",
        )

    # display(fh)
    plt.savefig(ofname.with_suffix(".pdf"))
    plt.savefig(
        ofname.with_suffix(".png"),
        transparent=True,
    )

    # fh = plt.figure(figsize=[10, 8])
    # ah = plt.axes()
    # # combined uncertainty estimate:
    # ah = self.binDat.plot(
    #     "Q",
    #     "I",
    #     yerr="IE",
    #     xerr="QE",
    #     logx=True,
    #     logy=True,
    #     zorder=1,
    #     color="red",
    #     ax=ah,
    #     lw=0.5,
    #     label="re-binned datapoints",
    #     fmt="-",
    #     alpha=1,
    # )
    # plt.xlabel("q (nm$^{-1}$)")
    # plt.ylabel("Scattering Cross-section (m$^{-1}$)")
    # plt.legend(loc=0)
    # plt.grid(b=True, which="major", color="black", linestyle="-", alpha=1)
    # plt.minorticks_on()
    # plt.grid(b=True, which="minor", color="black", linestyle=":", alpha=0.8)
    # display(fh)
    # plt.savefig(Path(self.workon, self.outputfilename.with_suffix(".pdf")))
    # plt.savefig(
    #     Path(self.workon, self.outputfilename.with_suffix(".png")), transparent=True
    # )
    return


if __name__ == "__main__":
    # process input arguments
    parser = argparse.ArgumentParser(
        description="""
            Runs a datamerge binning/rebinning operation from the command line for processed MOUSE data. 
            For this to work, you need to have YAML-formatted configuration files ready. 

            Examples of these configuration files are provided in the examples subdirectory. 

            Released under a GPLv3+ license.
            """
    )
    # TODO: add info about output files to be created ...
    parser.add_argument(
        "-f",
        "--dataFiles",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "testdata" / "quickstartdemo1.csv",
        help="Path to the filenames with the SAXS data. If this is a directory, all *processed.nxs files are globbed",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--globKey",
        type=str,
        default="*processed.nxs",
        help="If filename path is a directory, this will be the glob key to find the files to merge",
        # required=True,
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "test.nxs",
        help="Path to the files to store the datamerge result in",
        # required=True,
    )
    parser.add_argument(
        "-C",
        "--configFile",
        type=lambda p: Path(p).absolute(),
        default=Path(__file__).absolute().parent / "defaults" / "mergeConfig.yaml",
        help="Path to the datamerge configuration (yaml) file",
        # required=True,
    )

    if isMac():
        # on OSX remove automatically provided PID,
        # otherwise argparse exits and the bundle start fails silently
        for i in range(len(sys.argv)):
            if sys.argv[i].startswith("-psn"):  # PID provided by osx
                del sys.argv[i]
    try:
        args = parser.parse_args()
    except SystemExit:
        raise

    # initiate logging (to console stderr for now)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    adict = vars(args)

    m = mergeCore(
        config=mergeConfigObjFromYaml(adict["configFile"]),
        dataList=getFiles(adict),
    )
    filteredMDO = m.run()
    # export to the final files
    ofname=Path(adict["outputFile"])
    logging.debug(f"8. Storing result in output file {ofname}")
    outputToNX(
        ofname=ofname, mco=m.config, mdo=filteredMDO, rangeList=m.ranges
    )
    # make the plots.
    plotFigure(m, ofname=Path(adict["outputFile"]))
