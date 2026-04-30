# Genome engineering using the CRISPR-Cas9 system

**Protocol by:** F Ann Ran, Patrick D Hsu, Jason Wright, Vineeta Agarwala, David A Scott, Feng Zhang

---

## MATERIALS

### REAGENTS

#### sgRNA preparation

* Plasmids: pSpCas9 (Addgene 48137), pSpCas9(BB) (Addgene 42230), pSpCas9(BB)-2A-GFP (Addgene 48138), pSpCas9(BB)-2A-Puro (Addgene 48139), pSpCas9n(BB) (Addgene 48873), pSpCas9n(BB)-2A-GFP (Addgene 48140), pSpCas9n(BB)-2A-Puro (Addgene 48141)
* pUC19 (Invitrogen, cat. no. 15364-011)
* PCR primers or oligos for sgRNA construction
* UltraPure DNase/RNase-free distilled water
* Herculase II fusion polymerase with 5× reaction buffer (Agilent Technologies, cat. no. 600679)
* Taq DNA polymerase with standard Taq buffer (NEB, cat. no. M0273S)
* dNTP solution mix, 25 mM each
* MgCl2, 25 mM
* QIAquick gel extraction kit
* QIAprep spin miniprep kit
* UltraPure TBE buffer, 10×
* SeaKem LE agarose
* SYBR Safe DNA stain, 10,000×
* 1-kb Plus DNA ladder
* TrackIt CyanOrange loading buffer
* FastDigest BbsI (BpiI)
* Fermentas Tango buffer
* DTT
* T7 DNA ligase with 2× rapid ligation buffer
* T4 polynucleotide kinase
* T4 DNA ligase reaction buffer, 10×
* Adenosine 5′-triphosphate, 10 mM
* PlasmidSafe ATP-dependent DNase
* One Shot Stbl3 chemically competent *E. coli*
* SOC medium
* LB medium
* LB agar medium
* Ampicillin, 100 mg ml–1, sterile filtered

#### Mammalian cell culture

* HEK 293FT cells
* HUES 9 cell line
* DMEM, high glucose
* DMEM, high glucose, no phenol red
* Dulbecco's PBS (DPBS)
* FBS, qualified and heat inactivated
* Opti-MEM I reduced-serum medium
* Penicillin-streptomycin, 100×
* Puromycin dihydrochloride
* TrypLE Express, no phenol red
* Lipofectamine 2000 transfection reagent
* Amaxa SF cell line 4D-Nucleofector X kit S, 32 RCT
* Geltrex LDEV-free reduced growth factor basement membrane matrix
* mTeSR1 medium
* Normocin
* Accutase cell detachment solution
* ROCK inhibitor (Y-27632)
* Amaxa P3 primary cell 4D-Nucleofector X kit S, 32 RCT

#### Genotyping analysis

* PCR primers for SURVEYOR, RFLP analysis or sequencing
* QuickExtract DNA extraction solution
* SURVEYOR mutation detection kit
* TBE Gels, 4–20%, 1.0 mm, 15 well
* Novex Hi-Density TBE sample buffer, 5×
* SYBR Gold nucleic acid gel stain, 10,000×
* FastDigest HindIII
* FastDigest buffer, 10×
* FastAP Antarctic phosphatase
* Nextera XT index kit

---

## PROCEDURE

### Design of targeting components and the use of the CRISPR Design tool ● TIMING 1 d

1. **Input target genomic DNA sequence.** Use the online CRISPR Design Tool (<http://tools.genome-engineering.org>) to input a sequence (e.g., 1-kb genomic fragment), identify and rank suitable target sites, and computationally predict off-target sites. Alternatively, manually select guide sequences by identifying the 20-bp sequence directly upstream of any 5′-NGG.

2. Order necessary oligos and primers as specified by the online tool.

### Design of the ssODN template (optional) ● TIMING 1 h

3. **Design and order custom ssODN.** Purchase either sense or antisense ssODN. Recommend homology arms of at least 40 nt on either side, preferably 90 nt for optimal HDR efficiency. PAGE purification is not necessary.

4. Resuspend and dilute ssODN ultramers to 10 μM final concentration. Store at –20 °C.

### Preparation of sgRNA expression construct

5. Generate the sgRNA expression construct using either:

**(A) PCR amplification ● TIMING 2 h**

i. **Preparation of diluted U6 PCR template.** Dilute pSpCas9(BB) or pSpCas9n(BB) template to 10 ng μl–1.

ii. **Preparation of diluted PCR primers.** Dilute U6-Fwd and U6-Rev primers to 10 μM.

iii. **U6-sgRNA PCR.** Set up reaction:
- Herculase II PCR buffer, 5×: 10 μl (1× final)
- dNTP, 100 mM: 0.5 μl (1 mM final)
- U6 PCR template: 1 μl (0.2 ng μl−1 final)
- U6-Fwd primer: 1 μl (0.2 μM final)
- U6-Rev primer: 1 μl (0.2 μM final)
- Herculase II fusion polymerase: 0.5 μl
- Distilled water: 36 μl
- Total: 50 μl

iv. Perform PCR:
- 95 °C, 2 min
- 30 cycles: 95 °C, 20 s; 60 °C, 20 s; 72 °C, 20 s
- 72 °C, 3 min

v. Run 5 μl on 2% agarose gel to verify 370-bp product.

vi. Purify PCR product using QIAquick PCR purification kit. Elute in 35 μl EB buffer.

**(B) Cloning sgRNA into pSpCas9(BB) vector ● TIMING 3 d**

i. **Preparation of sgRNA oligos inserts.** Resuspend top and bottom strand oligos to 100 μM. Prepare mixture:
- sgRNA top (100 μM): 1 μl
- sgRNA bottom (100 μM): 1 μl
- T4 ligation buffer, 10×: 1 μl
- T4 PNK: 1 μl
- ddH2O: 6 μl
- Total: 10 μl

ii. Phosphorylate and anneal: 37 °C for 30 min; 95 °C for 5 min; ramp down to 25 °C at 5 °C min–1.

iii. Dilute phosphorylated and annealed oligos 1:200.

iv. **Cloning the sgRNA oligos into pSpCas9(BB).** Set up ligation:
- pSpCas9(BB), 100 ng: × μl
- Diluted oligo duplex: 2 μl
- Tango buffer, 10×: 2 μl
- DTT, 10 mM: 1 μl
- ATP, 10 mM: 1 μl
- FastDigest BbsI: 1 μl
- T7 ligase: 0.5 μl
- ddH2O: to 20 μl
- Total: 20 μl

v. Incubate: 6 cycles of 37 °C for 5 min, 21 °C for 5 min.

vi. **PlasmidSafe treatment:**
- Ligation reaction: 11 μl
- PlasmidSafe buffer, 10×: 1.5 μl
- ATP, 10 mM: 1.5 μl
- PlasmidSafe exonuclease: 1 μl
- Total: 15 μl

vii. Incubate at 37 °C for 30 min, then 70 °C for 30 min.

viii. **Transformation.** Transform 2 μl into Stbl3 competent cells. Plate on LB-ampicillin plates. Incubate overnight at 37 °C.

ix. Day 2: Inspect plates for colony growth.

x. Pick 2-3 colonies into 3-ml LB-ampicillin cultures. Incubate overnight at 37 °C.

xi. Day 3: Isolate plasmid DNA using QIAprep spin miniprep kit.

xii. **Sequence validation.** Verify sequence using U6-Fwd primer.

### Functional validation of sgRNAs: HEK 293FT cell culture and transfections ● TIMING 3–4 d

6. **HEK 293FT maintenance.** Culture cells in D10 medium at 37 °C and 5% CO2.

7. **Passage cells.** Remove medium, rinse with DPBS, add 2 ml TrypLE to T75 flask, incubate 5 min at 37 °C. Add 10 ml D10 medium, transfer to 50-ml tube, dissociate by pipetting, reseed into new flasks.

8. **Preparation of cells for transfection.** Plate cells onto 24-well plates at 1.3 × 105 cells per well in 500 μl D10 medium without antibiotics, 16–24 h before transfection.

9. **Transfection.** On day of transfection, cells should be 70–90% confluent. Transfect using Lipofectamine 2000 or Amaxa SF kit:
- For pSpCas9(sgRNA): transfect 500 ng
- For PCR-based: mix 400 ng pSpCas9 + 20 ng sgRNA amplicon + pUC19 to 500 ng total

10. Add Lipofectamine complex gently to cells.

11. Check cells after 24 h for transfection efficiency (>70% expected).

12. Supplement culture with 500 μl warm D10 medium.

13. Incubate for 48–72 h total after transfection before harvesting for analysis.

### Co-transfection of CRISPR plasmids and HDR templates into HEK 293FT cells (optional) ● TIMING 3–4 d

14. Linearize 1–2 μg targeting vector if using plasmid. For ssODNs, resuspend to 10 μM and skip to Step 17.

15. Run sample on gel to verify linearization.

16. Purify linearized plasmid with QIAQuick PCR Purification kit, elute in 35 μl EB buffer.

17. **Preparation of cells for transfection.** Culture HEK 293FT in T75 or T225 flasks. Plan for 2 × 105 cells per transfection.

18. **Prewarming plates.** Add 1 ml warm D10 medium to each well of 12-well plate. Place in incubator.

19. **Pre-mix DNA:**
- **(A) For HDR-targeting plasmid:** 500 ng Cas9 plasmid + 500 ng linearized targeting plasmid
- **(B) For ssODN:** 500 ng Cas9 plasmid + 1 μl ssODN (10 μM)

20. **Dissociation of cells.** Remove medium, rinse with DPBS, add 2 ml TrypLE, incubate 5 min at 37 °C, add 10 ml D10 medium, triturate gently.

21. Count cells. Calculate volume needed for 2 × 105 cells per transfection (plus 20% extra).

22. Spin down cells at 200g for 5 min at room temperature.

23. Prepare transfection solution: mix SF solution and S1 supplement (20 μl per transfection).

24. Resuspend cells in S1-supplemented SF solution (20 μl per 2 × 105 cells).

25. Pipette 20 μl resuspended cells into each DNA premix. Transfer to Nucleocuvette strip.

26. Electroporate using program CM-130.

27. Gently pipette 100 μl warm D10 medium into each Nucleocuvette, transfer to prewarmed well.

28. Incubate 24 h. Check transfection efficiency (>70–80% expected).

29. Add 1 ml warm D10 medium. Apply puromycin selection (1–3 μg ml–1) if desired. Incubate for at least 72 h.

### hESC (HUES 9) culture and transfection ● TIMING 3–4 d

30. **Maintaining HUES9 cells.** Maintain in feeder-free conditions with mTesR1 medium supplemented with 100 μg ml–1 Normocin.

31. Prepare 10-ml aliquot of mTeSR1 medium with 10 μM ROCK inhibitor.

32. **Coating tissue culture plate.** Dilute cold GelTrex 1:100 in cold DMEM, coat 100-mm plate.

33. Incubate plate at 37 °C for at least 30 min.

34. Thaw vial of cells at 37 °C, transfer to 15-ml tube, add 5 ml mTeSR1, pellet at 200g for 5 min.

35. Aspirate GelTrex coating, seed ~1 × 106 cells with 10 ml mTeSR1 + ROCK inhibitor.

36. Replace with mTeSR1 without ROCK inhibitor after 24 h. Refeed daily.

37. **Passaging cells.** Passage before 70% confluency.

38. Aspirate medium, wash with DPBS.

39. Add 2 ml Accutase, incubate at 37 °C for 3–5 min.

40. Add 10 ml mTeSR1, transfer to 15-ml tube, resuspend gently.

41. Replate onto GelTrex-coated plates in mTeSR1 + 10 μM ROCK inhibitor.

42. Replace with normal mTeSR1 after 24 h.

43. **Transfection.** Culture cells for at least 1 week after thawing before transfecting.

44. Refeed log-phase cells (50–70% confluency) 2 h before transfection.

45. Dissociate to single cells or small clusters with Accutase and gentle resuspension.

46. Count cells (200,000 per transfection), spin down at 200g for 5 min.

47. Resuspend in 20 μl S1-supplemented P3 nucleofection solution per 2 × 105 cells.

48. Add DNA (1 μg total), pipette into electroporation cuvettes, electroporate.

49. Plate electroporated cells onto coated 100-mm plates with 10 μM ROCK inhibitor.

50. Check transfection success (>70% expected). Refeed daily with mTeSR1 beginning 24 h after nucleofection. Apply puromycin selection (0.5 μg ml–1) if desired.

51. At 48–72 h post transfection, dissociate cells with Accutase, resuspend in 5× volume mTeSR1. Reserve fraction for replating, downstream applications, or clonal isolation. Use remaining for genotyping.

52. Spin cells at 200g for 5 min.

53. Process pelleted cells for DNA extraction.

### Isolation of clonal cell lines by FACS ● TIMING 2–3 h hands-on; 2–3 weeks expansion

54. **Preparation of FACS media.** Filter D10 medium with penicillin-streptomycin through 0.22-μM filter.

55. Add 100 μl D10 + penicillin-streptomycin per well to 96-well plates.

56. **Preparation of cells for FACS.** Dissociate cells by aspirating medium, adding TrypLE, incubating 5 min at 37 °C, adding 400 μl D10 medium.

57. Transfer to 15-ml tube, triturate 20 times.

58. Spin at 200g for 5 min.

59. Resuspend in 200 μl FACS medium.

60. Filter cells through cell strainer tube. Place on ice until sorting.

61. Sort single cells into 96-well plates. Examine plate under microscope to confirm single cells.

62. Expand cells for 2–3 weeks. Add 100 μl warm D10 medium 5 d after sorting. Change 100 μl medium every 3–5 d.

63. Inspect colonies 1 week after sorting for clonal appearance. Mark empty wells or wells with multiple colonies.

64. When >60% confluent, prepare replica plates. Dissociate by vigorous pipetting, plate 20% into replica wells. Change medium every 2–3 d.

65. Use remaining 80% for DNA isolation and genotyping.

### Isolation of clonal cell lines by dilution ● TIMING 2–3 h hands-on; 2–3 weeks expansion

66. Dissociate cells 48 h after transfection. Ensure single-cell dissociation.

67. Count cells, serially dilute to 0.5 cells per 100 μl (60 cells in 12 ml D10 per 96-well plate). Plate at least two 96-well plates per transfected population.

68. Multichannel-pipette 100 μl diluted cells per well.

69. Inspect colonies ~1 week after plating for clonal appearance. Mark wells with multiple colonies.

70. Expand for 2–3 weeks. Refeed and replica-plate as needed.

### Functional testing: detection of indel mutations by SURVEYOR nuclease assay ● TIMING 5–6 h

71. **Harvesting cells for DNA extraction.** Dissociate transfected cells, spin at 200g for 5 min.

72. Aspirate medium completely.

73. Extract DNA using QuickExtract solution (50 μl for 24-well, 10 μl for 96-well).

74. Normalize extracted DNA to 100–200 ng μl–1 with ddH2O.

75. **Setting up SURVEYOR PCR.** Master-mix:
- Herculase II PCR buffer, 5×: 10 μl (1× final)
- dNTP, 100 mM: 1 μl (2 mM final)
- SURVEYOR-Fwd primer, 10 μM: 1 μl (0.2 μM final)
- SURVEYOR-Rev primer, 10 μM: 1 μl (0.2 μM final)
- Herculase II fusion polymerase: 1 μl
- MgCl2, 25 mM: 2 μl (1 mM final)
- DNA template: 1 μl (2 ng μl−1 final)
- ddH2O: 33 μl
- Total: 50 μl

76. Perform PCR (no more than 30 cycles):
- 95 °C, 2 min
- 30 cycles: 95 °C, 20 s; 60 °C, 20 s; 72 °C, 30 s
- 72 °C, 3 min

77. Run 2–5 μl on 1% agarose gel to check for single-band products.

78. Purify PCRs with QIAQuick PCR purification kit, normalize to 20 ng μl–1.

79. **DNA heteroduplex formation.** Set up annealing:
- Taq PCR buffer, 10×: 2 μl
- Normalized PCR product, 20 ng μl−1: 18 μl
- Total: 20 μl

80. Anneal using temperature stepping program (95 °C to 4 °C with gradual cooling).

81. **SURVEYOR nuclease S digestion.** Add to annealed heteroduplexes:
- Annealed heteroduplex: 20 μl
- MgCl2 stock, 0.15 M: 2.5 μl (15 mM final)
- ddH2O: 0.5 μl
- SURVEYOR nuclease S: 1 μl (1× final)
- SURVEYOR enhancer S: 1 μl (1× final)
- Total: 25 μl

82. Vortex, spin down, incubate at 42 °C for 30 min.

83. (Optional) Add 2 μl Stop Solution.

84. **Visualizing SURVEYOR reaction.** Run 10 μl on 2% agarose gel or 4–20% gradient polyacrylamide TBE gel.

85. Stain gel with SYBR Gold (1:10,000 in TBE) for 15 min.

86. Image gel using quantitative imaging system.

87. **Estimation of cleavage intensity.** Measure integrated intensity of PCR amplicon and cleaved bands.

88. Calculate fraction cleaved: *fcut* = (*b + c*)/(*a + b + c*), where *a* = undigested product, *b* and *c* = cleavage products.

89. Estimate indel occurrence: indel(%) = 100 × (1 − √(1 − *fcut*))

### Functional testing: detection of genomic microdeletions by PCR ● TIMING 3–4 h hands-on; 2–3 weeks expansion

90. Transfect cells with pair of sgRNAs flanking region to be deleted.

91. At 24 h after transfection, isolate clones by FACS or serial dilution.

92. Expand cells for 2–3 weeks.

93. Extract DNA, normalize to 100 ng μl–1.

94. **PCR amplification and analysis:**

**(A) Deletion or microdeletion analysis**

i. Use Out-Fwd and Out-Rev primers to verify deletion by product size. Set up PCR:
- Herculase II PCR buffer, 5×: 10 μl (1× final)
- dNTP, 100 mM: 1 μl (2 mM final)
- Out-Fwd primer, 10 μM: 1 μl (0.2 μM final)
- Out-Rev primer, 10 μM: 1 μl (0.2 μM final)
- Herculase II fusion polymerase: 1 μl
- MgCl2, 25 mM: 2 μl (1 mM final)
- DNA template: 1 μl (2 ng μl−1 final)
- ddH2O: 33 μl
- Total: 50 μl

**(B) Inversion analysis**

i. Set up PCR with Out-Fwd + In-Fwd or Out-Rev + In-Rev primer pairs (same reaction composition as above).

95. Perform PCR:
- 95 °C, 2 min
- 30 cycles: 95 °C, 20 s; 60 °C, 20 s; 72 °C, 30 s
- 72 °C, 3 min

96. Run 2–5 μl on 1–2% agarose gel to check product size (deletions) or presence/absence (inversions).

### Functional testing: genotyping of HDR-mediated targeted modifications by RFLP analysis ● TIMING 3–4 h

97. Extract DNA, normalize to 100–200 ng μl–1.

98. **PCR amplification of modified region.** Use HDR-Fwd and HDR-Rev primers:
- Herculase II PCR buffer, 5×: 10 μl (1× final)
- dNTP, 100 mM: 1 μl (2 mM final)
- HDR-Fwd primer, 10 μM: 1 μl (0.2 μM final)
- HDR-Rev primer, 10 μM: 1 μl (0.2 μM final)
- Herculase II fusion polymerase: 1 μl
- MgCl2, 25 mM: 2 μl (1 mM final)
- DNA template: 1 μl (2 ng μl−1 final)
- ddH2O: 33 μl
- Total: 50 μl

99. Run PCR:
- 95 °C, 2 min
- 30 cycles: 95 °C, 20 s; 60 °C, 20 s; 72 °C, 30-60 s per kb
- 72 °C, 3 min

100. Run 5 μl on 0.8–1% agarose gel to check for single band.

101. Purify PCRs using QIAquick PCR purification kit.

102. **RFLP analysis.** Digest with appropriate restriction enzyme:
- Purified PCR amplicon: × μl (200-300 ng)
- FastDigest buffer: 1 μl
- HindIII (or other enzyme): 0.5 μl
- ddH2O: to 10 μl
- Total: 10 μl

103. Digest for 10 min at 37 °C.

104. Run 10 μl on 4–20% gradient polyacrylamide TBE gel.

105. Stain with SYBR Gold for 15 min.

106. Image and quantify cleavage products.

107. Estimate HDR efficiency: (*b + c*)/(*a + b + c*), where *a* = undigested product, *b* and *c* = cut fragments.

108. Alternatively, clone and sequence PCR amplicons (Steps 109–117) or perform deep sequencing (Steps 118–126).

### Assessment of Cas9 cleavage or HDR-mediated target modification efficiency by Sanger sequencing ● TIMING 3 d

109. **Target-site amplicon digestion:**
- FastDigest buffer, 10×: 3 μl
- FastDigest EcoRI: 1 μl
- FastDigest HindIII: 1 μl
- Purified PCR product, 20 ng μl−1: 10 μl
- ddH2O: 15 μl
- Total: 30 μl

110. **pUC19 backbone digestion:**
- FastDigest buffer, 10×: 3 μl
- FastDigest EcoRI: 1 μl
- FastDigest HindIII: 1 μl
- FastAP alkaline phosphatase: 1 μl
- pUC19 vector (200 ng μl−1): 5 μl
- ddH2O: 20 μl
- Total: 30 μl

Incubate at 37 °C for 15 min.

111. Purify digestion reactions with QIAQuick PCR purification kit.

112. **Ligation.** Ligate at 1:3 vector:insert ratio:
- Digested pUC19: × μl (50 ng)
- Digested PCR product: × μl (1:3 molar ratio)
- T7 Ligase: 1 μl
- Rapid Ligation buffer, 2×: 10 μl
- ddH2O: to 20 μl
- Total: 20 μl

Incubate at room temperature for 15 min.

113. **PlasmidSafe treatment:**
- Ligation reaction: 11 μl
- PlasmidSafe buffer, 10×: 1.5 μl
- ATP, 10 mM: 1.5 μl
- PlasmidSafe exonuclease: 1 μl
- Total: 15 μl

114. **Transformation.** Transform 5 μl into Stbl3 cells. Plate on LB-ampicillin plates. Incubate overnight at 37 °C.

115. Day 2: Pick minimum of 48 clones into 3 ml LB-ampicillin cultures.

116. Day 3: Isolate plasmid DNA using QIAprep spin miniprep kit.

117. **Sanger sequencing.** Sequence using pUC19-Fwd or pUC19-Rev primer. Calculate editing efficiency as (no. of modified clones)/(no. of total clones).

### Deep sequencing and off-target analysis ● TIMING 2–3 d

118. **Designing deep-sequencing primers.** Design primers for 100- to 200-bp amplicons using NCBI Primer-Blast or CRISPR Design Tool.

119. Extract genomic DNA, normalize to 100–200 ng μl–1.

120. **Initial library preparation-PCR:**
- Herculase II PCR buffer, 5×: 10 μl (1× final)
- dNTP, 100 mM: 1 μl (2 mM final)
- Fwd primer (10 μM): 1 μl (0.2 μM final)
- Rev primer (10 μM): 1 μl (0.2 μM final)
- Herculase II fusion polymerase: 1 μl
- MgCl2 (25 mM): 2 μl (1 mM final)
- DNA template: 1 μl (2 ng μl−1 final)
- ddH2O: 33 μl
- Total: 50 μl

121. Perform PCR (no more than 20 cycles):
- 95 °C, 2 min
- 20 cycles: 95 °C, 20 s; 60 °C, 20 s; 72 °C, 15 s
- 72 °C, 3 min

122. Run 2–5 μl on 1% agarose gel to check for single-band product.

123. Purify PCRs using QIAQuick PCR purification kit, normalize to 20 ng μl–1.

124. **Nextera XT DNA sample preparation.** Generate Miseq sequencing-ready libraries with unique bar codes according to manufacturer's protocol.

125. Sequence samples on Illumina Miseq according to user manual.

126. **Analyze sequencing data.** Perform indel analysis using read alignment programs (ClustalW, Geneious) or sequence analysis scripts.