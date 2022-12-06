#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from .mergecore import mergeCore
import logging


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

        ah.text(
            x=drange.scatteringData.qMin() * 1.1,
            y=ystart * 1.4,
            s=f"Range {drange.rangeId} x {drange.scale:0.04f}",
        )

    # display(fh)
    if ofname is not None:
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
