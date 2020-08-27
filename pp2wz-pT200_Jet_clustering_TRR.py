#!/usr/bin/env python
# 
# Translation + Rotation + Reflection
# Translation : Leading subjet, put highest pT in leading subjet at origin
# Rotation : subleading subjet, put highest pT in subleading subjet blow the origin
# Reflection : if second subjet exists, sum over intensities of pT left&right side, put higher intensity on right side
#              if third subjet exists, third subjet goes on the right side


import sys
import math
import os
from array import array
import pandas as pd
import ROOT
from pyjet import cluster
from ROOT import TLorentzVector, TH1F, TCanvas
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import LogNorm

try:
    input = raw_input
except:
    pass

if len(sys.argv) < 4:
    print(" Usage: python Jet_clustering-multi-run-pTextract_TRR.py run_location run_number segment_number")
    sys.exit(1)

#Example :  python examples/Jet_clustering-2subjet_noRefl-multi-run.py /Volumes/Tae_Research-Backup/pp2wjet_run/

#Load Delphes (Code should be inside of Delphes directory or under Delphes/examples directory)
ROOT.gSystem.Load("libDelphes")

try:
    ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
    ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass


# Create chain of root trees, 20 runs
chain = ROOT.TChain("Delphes")
#for i in range (1,21):
#    inputfile = sys.argv[1]+"run_"+str(i)+"/*delphes_events.root"
#    chain.Add(inputfile)

inputfile = sys.argv[1]+"run_"+sys.argv[2]+"/*delphes_events.root"
chain.Add(inputfile)


# Create object of class ExRootTreeReader
treeReader = ROOT.ExRootTreeReader(chain)
numberOfEntries = treeReader.GetEntries()
#numberOfEntries = 100 # Debug and development purpose -Tae-

wjpTmin = 200
wjpTmax = 300

#sys.argv[3] from 0 to 4
init = int(sys.argv[3])*10000
fin = (int(sys.argv[3])+1)*10000
if sys.argv[3]==4:
    fin = numberOfEntries

print "Event from "+str(init)+" to "+str(fin)

# Get pointers to branches used in this analysis
branchJet = treeReader.UseBranch("Jet")
branchTrack = treeReader.UseBranch("EFlowTrack")
branchPhoton = treeReader.UseBranch("EFlowPhoton")
branchNeutral = treeReader.UseBranch("EFlowNeutralHadron")
branchParticle = treeReader.UseBranch("Particle")

jet_cluster = pd.DataFrame()
jet_result = pd.DataFrame()
jet_TR_event = pd.DataFrame()
event_jet_cent = pd.DataFrame()


# Redefine event number for 2 subjet events
event_val = 1

clstr_val = 0
reclstr_val = 0
morethan2subjet = 0

# Loop over all events
for entry in range(init, fin):
    
    # Load selected branches with data from specified event
    treeReader.ReadEntry(entry)
    emptydf = pd.DataFrame()
    vectors = []
    
    temp = TLorentzVector()
    wpart = TLorentzVector()

    #NOTE: for pin in range(0, branchParticle.GetEntriesFast()):# (slower) Don't need since W is located close to the beginning
    # Check if it is pp->h->ww or pp->w jet event
    # For array of PID, check if higgs PID exists. Then process differently
    pids = []
    
    for pid in range(0, 20):
        part = branchParticle.At(pid)
        pids.append(part.PID)
    
    pids = np.array(pids)
    #print "Total pid : "+str(ppids.shape[0])
    #for i in range(ppids.shape[0]):
    #    if ppids[i]==24:
    #        print i
    
    
    #print pids[:20]
    #print entry
    # Higgs id = 25
    if 25 in pids:
        print "Higgs"
    
    # No higgs event
    else:
        #print "No Higgs!"
        for pin in range(0, branchParticle.GetEntriesFast()):
        #for pin in range(0, 20):
            part = branchParticle.At(pin)
            if (abs(part.PID) == 24 and part.Status == 22):
                wpart.SetPtEtaPhiE(part.PT, part.Eta, part.Phi, part.E)
    
    
    for etrenrty in range(0, branchTrack.GetEntriesFast()):
        eterm = branchTrack.At(etrenrty)
        temp.SetPtEtaPhiE(eterm.PT, eterm.Eta, eterm.Phi, eterm.PT*np.cosh(eterm.Eta))
        vectors.append((temp.E(), temp.Px(), temp.Py(), temp.Pz()))

    for pentry in range(0, branchPhoton.GetEntriesFast()):
        pterm = branchPhoton.At(pentry)
        temp.SetPtEtaPhiE(pterm.ET, pterm.Eta, pterm.Phi, pterm.E)
        vectors.append((temp.E(), temp.Px(), temp.Py(), temp.Pz()))

    for nentry in range(0, branchNeutral.GetEntriesFast()):
        nterm = branchNeutral.At(nentry)
        temp.SetPtEtaPhiE(nterm.ET, nterm.Eta, nterm.Phi, nterm.E)
        vectors.append((temp.E(), temp.Px(), temp.Py(), temp.Pz()))
    
    vects = np.asarray(vectors, dtype=np.dtype([('E', 'f8'), ('px', 'f8'), ('py', 'f8'), ('pz', 'f8')]) )
    
    # Jet clustering
    sequence = cluster(vects,R=1.0,p=-1,ep=True) # antikt: p = -1 , Cambridge/Aachen: p = 0, kt: p = 1
    jets = sequence.inclusive_jets(ptmin=100.0)
    
    if not (not jets):
        clstr_val = clstr_val + 1
    
    gotawjet = -1
    if len(jets) == 1:
        jeta = TLorentzVector(jets[0].px, jets[0].py, jets[0].pz, jets[0].e)
        dra = jeta.DeltaR(wpart) # angular distance between w-particle and jet position
        if (dra <= 0.5):
            wjet = jeta
            gotawjet = 0
        elif (dra > 0.5):
            pass
    elif len(jets) == 2:
        jeta = TLorentzVector(jets[0].px, jets[0].py, jets[0].pz, jets[0].e)
        jetb = TLorentzVector(jets[1].px, jets[1].py, jets[1].pz, jets[1].e)
        dra = jeta.DeltaR(wpart)
        drb = jetb.DeltaR(wpart)
        if (dra < drb and dra < 0.5):
            gotawjet = 0
        elif (drb <= 0.5):
            gotawjet = 1
        else:
            pass
    elif len(jets) == 3:
        jeta = TLorentzVector(jets[0].px, jets[0].py, jets[0].pz, jets[0].e)
        jetb = TLorentzVector(jets[1].px, jets[1].py, jets[1].pz, jets[1].e)
        jetc = TLorentzVector(jets[2].px, jets[2].py, jets[2].pz, jets[2].e)
        dra = jeta.DeltaR(wpart)
        drb = jetb.DeltaR(wpart)
        drc = jetc.DeltaR(wpart)
        #print min([dra, drb, drc])
        
        if (min([dra, drb, drc])==dra and dra < 0.5):
            gotawjet = 0
        elif (min([dra, drb, drc])==drb and drb < 0.5):
            gotawjet = 1
        elif (min([dra, drb, drc])==drc and drc < 0.5):
            gotawjet = 2
        else:
            pass
            
    
    elif len(jets) == 4:
        jeta = TLorentzVector(jets[0].px, jets[0].py, jets[0].pz, jets[0].e)
        jetb = TLorentzVector(jets[1].px, jets[1].py, jets[1].pz, jets[1].e)
        jetc = TLorentzVector(jets[2].px, jets[2].py, jets[2].pz, jets[2].e)
        jetd = TLorentzVector(jets[3].px, jets[3].py, jets[3].pz, jets[3].e)
        dra = jeta.DeltaR(wpart)
        drb = jetb.DeltaR(wpart)
        drc = jetc.DeltaR(wpart)
        drd = jetd.DeltaR(wpart)
        #print min([dra, drb, drc])
        
        if (min([dra, drb, drc, drd])==dra and dra < 0.5):
            gotawjet = 0
        elif (min([dra, drb, drc, drd])==drb and drb < 0.5):
            gotawjet = 1
        elif (min([dra, drb, drc, drd])==drc and drc < 0.5):
            gotawjet = 2
        elif (min([dra, drb, drc, drd])==drd and drd < 0.5):
            gotawjet = 3
        else:
            pass
        

    if gotawjet >= 0:
        wpx = jets[gotawjet].px
        wpy = jets[gotawjet].py
        wpt = np.sqrt(wpx**2+wpy**2)
        
        if wpt>=wjpTmin and wpt<=wjpTmax:
            wjet = jets[gotawjet]
            wjetconstits = wjet.constituents_array(ep=True)

            # Subjet clustering
            subjets_seq = cluster(wjetconstits, R=0.3,p=0, ep=True)
            subjets = subjets_seq.inclusive_jets(ptmin=1.)
            #subjets = subjets_seq.exclusive_jets(2)
            #print(subjets)

            # Put into dataframe and Assigning subjet number
            for jetent in range(len(subjets)):
                jet_constit = subjets[jetent].constituents_array()
                #print(subjets[jetent].constituents())

                for constit in range(len(subjets[jetent])):
                    jet_cluster = jet_cluster.append({'Event' : entry,
                                                      'jet number': jetent,
                                                      'jet constituent PT': jet_constit[constit][0],
                                                      'jet constituent Eta': jet_constit[constit][1],
                                                      'jet constituent Phi': jet_constit[constit][2],
                                                      'jet constituent Mass': jet_constit[constit][3]}, ignore_index=True)

            #print(jet_cluster)

            if not not subjets:
                reclstr_val = reclstr_val + 1

            if len(subjets) == 2:

                morethan2subjet = morethan2subjet + 1

                temp_jet = jet_cluster[jet_cluster['Event']==entry]
                temp_jet['jet px'] = temp_jet['jet constituent PT']*np.cos(temp_jet['jet constituent Phi'])
                temp_jet['jet py'] = temp_jet['jet constituent PT']*np.sin(temp_jet['jet constituent Phi'])
                temp_jet['jet pz'] = temp_jet['jet constituent PT']*np.sinh(temp_jet['jet constituent Eta'])


                # Translation
                jet1 = temp_jet[temp_jet['jet number']==0]
                pTmax1 = max(jet1['jet constituent PT'])
                eta_cent1 = jet1.loc[jet1['jet constituent PT']==pTmax1,'jet constituent Eta'].iat[0]
                phi_cent1 = jet1.loc[jet1['jet constituent PT']==pTmax1,'jet constituent Phi'].iat[0]

                jet_TR_event['Event'] = temp_jet['Event']
                jet_TR_event['jet number'] = temp_jet['jet number']
                jet_TR_event['jet constituent PT'] = temp_jet['jet constituent PT']
                jet_TR_event['jet trans Eta'] = temp_jet['jet constituent Eta']-eta_cent1
                jet_TR_event['jet trans Phi'] = temp_jet['jet constituent Phi']-phi_cent1
                jet_TR_event['jet px'] = temp_jet['jet px']
                jet_TR_event['jet py'] = temp_jet['jet py']
                jet_TR_event['jet pz'] = temp_jet['jet pz']
                #jet_TR_event['jet constituent Mass'] = temp_jet['jet constituent Mass']
                jet_TR_event['jet E'] = np.sqrt(temp_jet['jet constituent Mass']**2 + temp_jet['jet px']**2
                                                + temp_jet['jet py']**2 + temp_jet['jet pz']**2)


                # Rotation
                jet2 = jet_TR_event[jet_TR_event['jet number']==1]
                pTmax2 = max(jet2['jet constituent PT'])               
                eta_cent2 = jet2.loc[jet2['jet constituent PT']==pTmax2,'jet trans Eta'].iat[0]
                phi_cent2 = jet2.loc[jet2['jet constituent PT']==pTmax2,'jet trans Phi'].iat[0]
                angle = np.arctan2(eta_cent2, phi_cent2)
                cosangle = np.cos(3*np.pi/2-angle)
                sinangle = np.sin(3*np.pi/2-angle)

                jet_TR_event['Event redef'] = event_val
                jet_TR_event['Rotx'] = cosangle*jet_TR_event['jet trans Phi']-sinangle*jet_TR_event['jet trans Eta']
                jet_TR_event['Roty'] = sinangle*jet_TR_event['jet trans Phi']+cosangle*jet_TR_event['jet trans Eta']


                # Translation
                pTright = jet_TR_event[jet_TR_event['Rotx'] > 0]['jet constituent PT'].sum()
                pTleft = jet_TR_event[jet_TR_event['Rotx'] < 0]['jet constituent PT'].sum()

                #print("//===============================//")
                #print(jet_TR_event)
                #print(pTright , pTleft)


                if pTright < pTleft:
                    jet_TR_event['Qx'] = -1*jet_TR_event['Rotx']
                    jet_TR_event['Qy'] = jet_TR_event['Roty']
                    event_jet_cent = event_jet_cent.append(jet_TR_event)

                else:
                    jet_TR_event['Qx'] = jet_TR_event['Rotx']
                    jet_TR_event['Qy'] = jet_TR_event['Roty']
                    event_jet_cent = event_jet_cent.append(jet_TR_event)

                #print(jet_TR_event)
                #print("//===============================//")

                temp_jet = emptydf
                jet_TR_event = emptydf
                event_val = event_val + 1

                # Check initialization
                #print(temp_jet.empty,jet_TR_event.empty) 


            elif len(subjets) > 2:

                morethan2subjet = morethan2subjet + 1

                temp_jet = jet_cluster[jet_cluster['Event']==entry]
                temp_jet['jet px'] = temp_jet['jet constituent PT']*np.cos(temp_jet['jet constituent Phi'])
                temp_jet['jet py'] = temp_jet['jet constituent PT']*np.sin(temp_jet['jet constituent Phi'])
                temp_jet['jet pz'] = temp_jet['jet constituent PT']*np.sinh(temp_jet['jet constituent Eta'])


                # Translation
                jet1 = temp_jet[temp_jet['jet number']==0]
                pTmax1 = max(jet1['jet constituent PT'])
                eta_cent1 = jet1.loc[jet1['jet constituent PT']==pTmax1,'jet constituent Eta'].iat[0]
                phi_cent1 = jet1.loc[jet1['jet constituent PT']==pTmax1,'jet constituent Phi'].iat[0]

                jet_TR_event['Event'] = temp_jet['Event']
                jet_TR_event['jet number'] = temp_jet['jet number']
                jet_TR_event['jet constituent PT'] = temp_jet['jet constituent PT']
                jet_TR_event['jet trans Eta'] = temp_jet['jet constituent Eta']-eta_cent1
                jet_TR_event['jet trans Phi'] = temp_jet['jet constituent Phi']-phi_cent1
                jet_TR_event['jet px'] = temp_jet['jet px']
                jet_TR_event['jet py'] = temp_jet['jet py']
                jet_TR_event['jet pz'] = temp_jet['jet pz']
                #jet_TR_event['jet constituent Mass'] = temp_jet['jet constituent Mass']
                jet_TR_event['jet E'] = np.sqrt(temp_jet['jet constituent Mass']**2 + temp_jet['jet px']**2
                                                + temp_jet['jet py']**2 + temp_jet['jet pz']**2)


                # Rotation
                jet2 = jet_TR_event[jet_TR_event['jet number']==1]
                pTmax2 = max(jet2['jet constituent PT'])               
                eta_cent2 = jet2.loc[jet2['jet constituent PT']==pTmax2,'jet trans Eta'].iat[0]
                phi_cent2 = jet2.loc[jet2['jet constituent PT']==pTmax2,'jet trans Phi'].iat[0]
                angle = np.arctan2(eta_cent2, phi_cent2)
                cosangle = np.cos(3*np.pi/2-angle)
                sinangle = np.sin(3*np.pi/2-angle)

                jet_TR_event['Event redef'] = event_val
                jet_TR_event['Rotx'] = cosangle*jet_TR_event['jet trans Phi']-sinangle*jet_TR_event['jet trans Eta']
                jet_TR_event['Roty'] = sinangle*jet_TR_event['jet trans Phi']+cosangle*jet_TR_event['jet trans Eta']


                # Reflection
                jet3 = jet_TR_event[jet_TR_event['jet number']==2]
                pTmax3 = max(jet3['jet constituent PT'])
                Qx_cent = jet3.loc[jet3['jet constituent PT']==pTmax3,'Rotx'].iat[0]

                #print("//===============================//")
                #print(jet_TR_event)
                #print(Qx_cent)

                if Qx_cent < 0:
                    jet_TR_event['Qx'] = -1*jet_TR_event['Rotx']
                    jet_TR_event['Qy'] = jet_TR_event['Roty']
                    event_jet_cent = event_jet_cent.append(jet_TR_event)

                else:
                    jet_TR_event['Qx'] = jet_TR_event['Rotx']
                    jet_TR_event['Qy'] = jet_TR_event['Roty']
                    event_jet_cent = event_jet_cent.append(jet_TR_event)

                #print(jet_TR_event)
                #print("//===============================//")

                temp_jet = emptydf
                jet_TR_event = emptydf
                event_val = event_val + 1

    #        elif len(subjets) == 1:



            else:
                pass
        else:
            pass
    
    # Flag
    if entry%1000 == 0:
        print(str(entry)+" / "+str(fin)+" Completed")



print "Total Events : "+str(fin-init)
print "Total clustered events : "+str(clstr_val)
print "Went through reclustering : "+str(reclstr_val)
print "More than 2 subjet of reclustering : "+str(morethan2subjet)
        
        

# Final Results
jet_result['Event'] = event_jet_cent['Event redef']
jet_result['Subjet number'] = event_jet_cent['jet number'] # May include later if needed
jet_result['Constituent pT'] = event_jet_cent['jet constituent PT']
jet_result['Qx'] = event_jet_cent['Qx']
jet_result['Qy'] = event_jet_cent['Qy']
jet_result['jet px'] = event_jet_cent['jet px']
jet_result['jet py'] = event_jet_cent['jet py']
jet_result['jet pz'] = event_jet_cent['jet pz']
jet_result['jet E'] = event_jet_cent['jet E']


# Shows How many events have 2 or more subjets
print("\n")
print("Number of 2 or more subjet events  :  "+str(max(jet_result['Event'])))
print("\n")
perc = float(max(jet_result['Event']))/float(10000)*100
print("Precentage of 2 or more subjet events : "+str(perc)+"%")
print("\n")


#Save Final Results
jet_result.to_csv(path_or_buf=sys.argv[1]+"TRR_clustering/jet_result_higheff-run_"+sys.argv[2]+"-"+sys.argv[3]+".csv",index=False)


#input("Press Enter to continue...")

