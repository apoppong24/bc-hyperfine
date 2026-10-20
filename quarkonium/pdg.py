"""Experimental charmonium and bottomonium masses used for the refit.

Values in MeV from the PDG 2026 edition (F. Takahashi et al., Particle Data
Group, Int. J. Mod. Phys. A 41, 2630011 (2026); meson summary tables
https://pdg.lbl.gov/2026/tables/rpp2026-tab-mesons-c-cbar.pdf and
.../rpp2026-tab-mesons-b-bbar.pdf, and the particle listings for the
D-wave charmonia).  Where two uncertainties are listed they are combined in
quadrature.  Differences from PDG 2024 (used in earlier commits) are noted
inline; all are < 1 MeV and the bottomonium table is identical.


Labels use *our* convention: n = radial excitation index starting at 1,
so χ_c(1P) is "13P0" etc.  Only states below the open-flavour threshold
(DD̄ at 3739 MeV, BB̄ at 10559 MeV) are included in the fit, except
ψ(3770) and ψ3(3842)/ψ2(3823), which are narrow and conventionally fitted.
Υ(4S) is listed but excluded from the fit by default.
"""

CHARMONIUM = {
    "11S0": (2984.09, 0.30),    # η_c(1S)   (2024: 2984.1 ± 0.4)
    "13S1": (3096.900, 0.006),  # J/ψ
    "13P0": (3415.50, 0.19),    # χ_c0(1P)  (2024: 3414.71 ± 0.30)
    "13P1": (3510.67, 0.05),    # χ_c1(1P)
    "11P1": (3525.37, 0.14),    # h_c(1P)
    "13P2": (3556.17, 0.07),    # χ_c2(1P)
    "21S0": (3637.8, 0.6),      # η_c(2S)   (2024: 3637.7 ± 0.9)
    "23S1": (3686.097, 0.011),  # ψ(2S)
    "13D1": (3773.7, 0.7),      # ψ(3770)
    "13D2": (3823.51, 0.34),    # ψ2(3823)  (2024: 3823.5 ± 0.5)
    "13D3": (3842.71, 0.20),    # ψ3(3842)  single LHCb measurement
}

BOTTOMONIUM = {
    "11S0": (9398.7, 2.0),      # η_b(1S)
    "13S1": (9460.40, 0.10),    # Υ(1S)
    "13P0": (9859.44, 0.52),    # χ_b0(1P)
    "13P1": (9892.78, 0.40),    # χ_b1(1P)
    "11P1": (9899.3, 0.8),      # h_b(1P)
    "13P2": (9912.21, 0.40),    # χ_b2(1P)
    "21S0": (9999.0, 4.2),      # η_b(2S)  Belle 9999.0 ± 3.5 (+2.8/−1.9); OMITTED from the PDG summary table; asymmetric error symmetrised
    "23S1": (10023.4, 0.5),     # Υ(2S)
    "13D2": (10163.7, 1.4),     # Υ2(1D)
    "23P0": (10232.5, 0.64),    # χ_b0(2P)
    "23P1": (10255.46, 0.55),   # χ_b1(2P)
    "21P1": (10259.8, 1.2),     # h_b(2P)
    "23P2": (10268.65, 0.55),   # χ_b2(2P)
    "33S1": (10355.1, 0.5),     # Υ(3S)
    "33P1": (10513.4, 0.7),     # χ_b1(3P)
    "33P2": (10524.0, 0.8),     # χ_b2(3P)
}
BOTTOMONIUM_ABOVE_THRESHOLD = {"43S1": (10579.4, 1.2)}   # Υ(4S)

BC = {
    "11S0": (6274.47, 0.32),    # B_c+
    "21S0": (6871.2, 1.0),      # B_c(2S)+
    "13S1": (6339.0, 1.8),      # B_c*+  ATLAS 2026, arXiv:2605.16228: 64.5 ± 1.4 (stat) +1.0/−1.4 (syst) above B_c
}
# Hyperfine observables (MeV)
ATLAS_HYP_1S = (64.5, 1.8)      # m(B_c*+) - m(B_c+), ATLAS 2026 (asymmetric syst symmetrised)
CMS_HYP_DIFF = (29.1, 1.7)      # Delta_1S - Delta_2S, CMS 2019
LHCB_HYP_DIFF = 30.9            # Delta_1S - Delta_2S, LHCb 2019 (peak difference)

# Lattice QCD reference values, MeV
LATTICE_HYP_1S = (54.0, 3.0)    # M(B_c*) - M(B_c), HPQCD, Dowdall et al., PRD 86, 094510 (2012); Mathur et al. PRL 121, 202002 (2018): 55(3)
LATTICE_BCSTAR = (6330.0, 9.0)  # M(B_c*), Gregory et al., PRL 104, 022001 (2010)
LATTICE_F = {"cc": (394.7, 2.4),   # f_eta_c, Davies et al., PRD 82, 114504 (2010)
             "bb": (667.0, 6.0),   # f_eta_b, McNeile et al., PRD 86, 074503 (2012)
             "cb": (427.0, 6.0)}   # f_B_c,   McNeile et al., PRD 86, 074503 (2012)
