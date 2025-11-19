import os


def init_root(base_args):
    import ROOT
    libdir = os.path.dirname(os.path.abspath(__file__))
    # Shared objects expected next to your sources (adjust path if needed)
    ROOT.gSystem.AddDynamicPath(libdir)
    ROOT.gSystem.Load(os.path.join(libdir, "..", "..", "CobremsGeneration_cc.so"))
    ROOT.gSystem.Load(os.path.join(libdir, "..", "..", "rootvisuals_C.so"))

    ebeam, ibeam = base_args["ebeam"], base_args["ibeam"]
    ROOT.cobrems = ROOT.CobremsGeneration(ebeam, ibeam)
    ROOT.cobrems.setBeamErms(base_args["ebeamrms"])
    ROOT.cobrems.setBeamEmittance(base_args["emittance"])
    ROOT.cobrems.setCollimatorSpotrms(base_args["vspotrms"] * 1e-3)
    ROOT.cobrems.setCollimatorDistance(base_args["coldist"])
    ROOT.cobrems.setCollimatorDiameter(base_args["coldiam"] * 1e-3)
    ROOT.cobrems.setTargetCrystal("diamond")
    ROOT.cobrems.setTargetThickness(base_args["radthick"] * 1e-6)
    return ROOT
