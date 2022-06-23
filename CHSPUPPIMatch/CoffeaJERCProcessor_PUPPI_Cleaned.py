#CoffeaJERCProcessor.py

import copy
import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor



import awkward as ak
#from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector



manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

ptbins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 
        150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000 ])


etabins =   np.array([-4.889,  -4.716,  -4.538,  -4.363,  -4.191,  -4.013,  -3.839,  -3.664,  -3.489,
           -3.314,  -3.139,  -2.964,  -2.853,  -2.65,  -2.5,  -2.322,  -2.172,  -2.043,  -1.93,  -1.83,
           -1.74,  -1.653,  -1.566,  -1.479,  -1.392,  -1.305,  -1.218,  -1.131,  -1.044,  -0.957,  -0.879,
           -0.783,  -0.696,  -0.609,  -0.522,  -0.435,  -0.348,  -0.261,  -0.174,  -0.087,  0,  0.087,  0.174,
           0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,
           1.305,  1.392,  1.479,  1.566,  1.653,  1.74,  1.83,  1.93,  2.043,  2.172,  2.322,  2.5,  2.65,
           2.853,  2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,
           4.889, 5.191 ])




"""@ Extended from JERC Coffea code for Aliina
"""
class Processor(processor.ProcessorABC):
    def __init__(self):
        
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
       
        jetpt_axis = hist.Bin("pt", r"$p_T$", ptbins)
        ptresponse_axis = hist.Bin("ptresponse", "RECO / GEN response (raw)", 100, 0, 5)
        corrected_ptresponse_axis = hist.Bin("ptresponse", "RECO / GEN response (after JEC)", 100, 0, 5)
        CHSPUPPIptresponse_axis = hist.Bin("ptresponse", "CHS / PUPPI response (raw)", 100, 0, 5)
        CHSPUPPIcorrected_ptresponse_axis = hist.Bin("ptresponse", "CHS / PUPPI response (after JEC)", 100, 0, 5)
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        #jetpt_axis = hist.Bin("jetpt", r"Jet $p_{T}$ [GeV]", 50, 0, 5000)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", etabins)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)
        manual_axis = hist.Bin("jetp", r"Jet Momentum [GeV]", manual_bins)

        self._accumulator = processor.dict_accumulator({
            'GenJetCounts': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis),
            'GenJetCountsWithDZCut': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis),
            'GenJetCountsMatchedPUPPI': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis),
            'GenJetCountsMatchedCHS': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis),
            'ptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis),
            'corrected_ptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, corrected_ptresponse_axis),
            'CHSptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, ptresponse_axis),
            'CHScorrected_ptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, corrected_ptresponse_axis),
            'CHSPUPPIptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, CHSPUPPIptresponse_axis),
            'CHSPUPPIcorrected_ptresponse': hist.Hist("Counts", dataset_axis, jetpt_axis, jeteta_axis, CHSPUPPIcorrected_ptresponse_axis),

            'jetmass': hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            
            'jetpt':     hist.Hist("Counts", dataset_axis, jetpt_axis),
            'jeteta':    hist.Hist("Counts", dataset_axis, jeteta_axis),
            'jetphi':    hist.Hist("Counts", dataset_axis, jetphi_axis),
            
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })
        
        
        ext = extractor()
        ext.add_weight_sets([
            "* * Summer20UL18_V2_MC_L2Relative_AK4PFPuppi.txt",
        ])
        ext.finalize()
        
        jec_stack_names = ["Summer20UL18_V2_MC_L2Relative_AK4PFPuppi"]
        
        evaluator = ext.make_evaluator()
        
        print(evaluator)
        print(evaluator.keys())
        
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        ### more possibilities are available if you send in more pieces of the JEC stack
        # mc2016_ak8_jxform = JECStack(["more", "names", "of", "JEC parts"])
        
        self.corrector = FactorizedJetCorrector(
            #Summer20_UL18_MC_L1FastJet_AK4PFchs=evaluator['Summer20_UL18_MC_L1FastJet_AK4PFchs'],
            Summer20UL18_V2_MC_L2Relative_AK4PFPuppi=evaluator['Summer20UL18_V2_MC_L2Relative_AK4PFPuppi'],
        )
#         uncertainties = JetCorrectionUncertainty(
#             Summer20_UL18_MC_L1FastJet_AK4PFchs=evaluator['Summer20_UL18_MC_L1FastJet_AK4PFchs']
#         )


        self.name_map = jec_stack.blank_name_map
        self.name_map['JetPt'] = 'pt'
        self.name_map['JetMass'] = 'mass'
        self.name_map['JetEta'] = 'eta'
        self.name_map['JetA'] = 'area'
        self.name_map['ptGenJet'] = 'pt_gen'
        self.name_map['ptRaw'] = 'pt_raw'
        self.name_map['massRaw'] = 'mass_raw'
        self.name_map['Rho'] = 'rho'

        
        self.jet_factory = CorrectedJetsFactory(self.name_map, jec_stack)
        
        print(dir(evaluator))
        print()

            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        
        output = self.accumulator.identity()
        
        dataset = events.metadata['dataset']
     
 
        selectedEvents = events[
            (ak.num(events.GenJet) >= 2)
        ]



        events = selectedEvents
        events = events[np.absolute(events.GenVtx.z-events.PV.z)<0.2]

        # change to selectedEvents
        GenJets = events.GenJet[:,0:2]
        jets = events.JetPuppi 
        CHSjets = events.Jet 
        AllGenJetsPt = GenJets.pt
        AllGenJetsEta = GenJets.eta

        eventsWithDZCut = events[np.absolute(events.GenVtx.z-events.PV.z)<0.2]
        GenJetsWithDZCut = eventsWithDZCut.GenJet[:,0:2]
        AllGenJetsWithDZCutPt = GenJetsWithDZCut.pt
        AllGenJetsWithDZCutEta = GenJetsWithDZCut.eta

        
        jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
        jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
        jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]


        events_cache = events.caches[0]

        corrected_jets = self.jet_factory.build(jets, lazy_cache=events_cache)

        #PUPPI
        matchedJets = ak.cartesian([GenJets, corrected_jets])
        deltaR = matchedJets.slot0.delta_r(matchedJets.slot1)
        matchedJets = matchedJets[deltaR < 0.2]
        matchedJets = matchedJets[ak.num(matchedJets) > 0]

        rawjetpt = (1-matchedJets.slot1.rawFactor) * matchedJets.slot1.pt
        jetpt = matchedJets.slot1.pt
        jeteta = matchedJets.slot1.eta

        matched_genjetpt = matchedJets.slot0.pt
        matched_genjeteta = matchedJets.slot0.eta
        
        #ptresponse = jetpt / matched_genjetpt
        ptresponse = rawjetpt / matched_genjetpt
        corrected_ptresponse = jetpt / matched_genjetpt

        #CHS
        CHSmatchedJets = ak.cartesian([GenJets, CHSjets])
        CHSdeltaR = CHSmatchedJets.slot0.delta_r(CHSmatchedJets.slot1)
        CHSmatchedJets = CHSmatchedJets[CHSdeltaR < 0.2]
        CHSmatchedJets = CHSmatchedJets[ak.num(CHSmatchedJets) > 0]

        CHSrawjetpt = (1-CHSmatchedJets.slot1.rawFactor) * CHSmatchedJets.slot1.pt
        CHSjetpt = CHSmatchedJets.slot1.pt
        CHSjeteta = CHSmatchedJets.slot1.eta

        CHSmatched_genjetpt = CHSmatchedJets.slot0.pt
        CHSmatched_genjeteta = CHSmatchedJets.slot0.eta
        
        #ptresponse = jetpt / matched_genjetpt
        CHSptresponse = CHSrawjetpt / CHSmatched_genjetpt
        CHScorrected_ptresponse = CHSjetpt / CHSmatched_genjetpt

        #MatchBoth at the same time
        BothmatchedJets = ak.cartesian([GenJets, corrected_jets, CHSjets])
        BothdeltaROne = BothmatchedJets.slot0.delta_r(BothmatchedJets.slot1)
        BothmatchedJets = BothmatchedJets[BothdeltaROne < 0.2]
        BothdeltaRTwo = BothmatchedJets.slot0.delta_r(BothmatchedJets.slot2)
        BothmatchedJets = BothmatchedJets[BothdeltaRTwo < 0.2]
        #Replace by s.th. like ... BothmatchedJets = BothmatchedJets[ak.all(BothdeltaROne < 0.2,  BothdeltaRTwo < 0.2)]
        BothmatchedJets = BothmatchedJets[ak.num(BothmatchedJets) > 0]

        BothPUPPIrawjetpt = (1-BothmatchedJets.slot1.rawFactor) * BothmatchedJets.slot1.pt
        BothCHSrawjetpt = (1-BothmatchedJets.slot2.rawFactor) * BothmatchedJets.slot2.pt
        BothPUPPIjetpt = BothmatchedJets.slot1.pt
        BothPUPPIjeteta = BothmatchedJets.slot1.eta
        BothCHSjetpt = BothmatchedJets.slot2.pt
        BothCHSjeteta = BothmatchedJets.slot2.eta

        Bothmatched_genjetpt = BothmatchedJets.slot0.pt
        Bothmatched_genjeteta = BothmatchedJets.slot0.eta
        
        #ptresponse = jetpt / matched_genjetpt
        BothPUPPIptresponse = BothPUPPIrawjetpt / Bothmatched_genjetpt
        BothPUPPIcorrected_ptresponse = BothPUPPIjetpt / Bothmatched_genjetpt

        BothCHSptresponse = BothCHSrawjetpt / Bothmatched_genjetpt
        BothCHScorrected_ptresponse = BothCHSjetpt / Bothmatched_genjetpt

        CHSPUPPIptresponse = BothCHSrawjetpt / BothPUPPIrawjetpt
        CHSPUPPIcorrected_ptresponse = BothCHSjetpt / BothPUPPIjetpt
        

        
        
        output['jetpt'].fill(dataset = dataset, 
                        pt = ak.flatten(matched_genjetpt))

        output['jeteta'].fill(dataset = dataset, 
                        jeteta = ak.to_numpy(ak.flatten(jeteta), allow_missing=True))
       

        output['GenJetCounts'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(AllGenJetsPt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(AllGenJetsEta), allow_missing=True))
        output['GenJetCountsWithDZCut'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(AllGenJetsWithDZCutPt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(AllGenJetsWithDZCutEta), allow_missing=True))

        output['GenJetCountsMatchedPUPPI'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(matched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(matched_genjeteta), allow_missing=True))

        output['GenJetCountsMatchedCHS'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(CHSmatched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(CHSmatched_genjeteta), allow_missing=True))

        output['ptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(matched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(matched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(ptresponse), allow_missing=True))
        
        output['corrected_ptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(matched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(matched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(corrected_ptresponse), allow_missing=True))
        
        output['CHSptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(CHSmatched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(CHSmatched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(CHSptresponse), allow_missing=True))
        
        output['CHScorrected_ptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(CHSmatched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(CHSmatched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(CHScorrected_ptresponse), allow_missing=True))
        
        output['CHSPUPPIptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(Bothmatched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(Bothmatched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(CHSPUPPIptresponse), allow_missing=True))

        output['CHSPUPPIcorrected_ptresponse'].fill(dataset=dataset, pt=ak.to_numpy(ak.flatten(Bothmatched_genjetpt), allow_missing=True),
                        jeteta=ak.to_numpy(ak.flatten(Bothmatched_genjeteta), allow_missing=True),
                        ptresponse=ak.to_numpy(ak.flatten(CHSPUPPIcorrected_ptresponse), allow_missing=True))
        
        
        return output

    def postprocess(self, accumulator):
        return accumulator



