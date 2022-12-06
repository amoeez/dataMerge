from datetime import datetime
from pathlib import Path
import pytest
import datamerge as dm
import logging


def test_integral()->None:
    def filelistFromArgs(argDict:dict) -> list:
        """
        Takes the parsed command-line argument dictionary
        and returns the list of filenames
        """
        fnames = argDict["dataFiles"]
        if len(fnames) == 1:
            if fnames[0].is_dir():
                # glob the files from the globkey in Path
                fnames = sorted(fnames[0].glob(argDict["globKey"]))
                logging.info(f"Found the following files to merge: {fnames}")
        assert len(fnames) > 0, "length of filename list to merge is zero, cannot merge."
        assert isinstance(fnames, list)
        return fnames

    # normally supplied from the command line:
    adict = {
        'dataFiles': [Path('Y:/Measurements/SAXS002/processingCode/dataMerge/tests/data/20220925/autoproc/group_6')], 
        'globKey': '*processed.nxs', 
        'outputFile': Path('Y:/Measurements/SAXS002/processingCode/dataMerge/test.nxs'), 
        'configFile': Path('Y:/Measurements/SAXS002/processingCode/dataMerge/tests/mergeConfigExample.yaml')
        }
    dataList = dm.readersandwriters.SDOListFromFiles(filelistFromArgs(adict))
    m = dm.mergecore.mergeCore(
        config=dm.readersandwriters.mergeConfigObjFromYaml(adict["configFile"]),
        dataList=dataList,
    )
    filteredMDO = m.run()
    ofname = Path(adict["outputFile"])
    dm.readersandwriters.outputToNX(ofname=ofname, mco=m.config, mdo=filteredMDO, rangeList=m.ranges)
    dm.plotting.plotFigure(m, ofname=Path(adict["outputFile"]))
    return

def test_readers()->None:
    so = dm.readersandwriters.scatteringDataObjFromNX(
        Path(".")
        / "tests"
        / "data"
        / "20220925"
        / "autoproc"
        / "group_6"
        / "20220925_42_expanded_stacked_processed.nxs"
    )

    mco = dm.readersandwriters.mergeConfigObjFromYaml(Path(".", "tests", "mergeConfigExample.yaml"))
    print(mco)
    return

