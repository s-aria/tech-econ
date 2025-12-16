
# Inputs from you
SPP_CAPEX = 220_000_000_002.0   # £
annual_OM = 223_333_313.81       # £/year
discount_rate = 0.07
lifetime_years = 40
availability = 0.70

# Alt‑C with EBW (Table 6)
gross_MWe_ebw = 831.7
parasitic_MWe_ebw = 683.1
net_MWe_ebw = gross_MWe_ebw - parasitic_MWe_ebw  # 148.6 MWe

# Alt‑C CAPEX (Figure 23: 63.84% of SPP CAPEX)
capex_percent = 0.6384
ALT_C_CAPEX = SPP_CAPEX * capex_percent

# Capital Recovery Factor
CRF = discount_rate * (1+discount_rate)**lifetime_years / ((1+discount_rate)**lifetime_years - 1)

# Annual net energy (MWh)
annual_net_MWh = net_MWe_ebw * 8760.0 * availability

# LCOE (£/MWh)
LCOE = (ALT_C_CAPEX * CRF + annual_OM) / annual_net_MWh
print(ALT_C_CAPEX, net_MWe_ebw, annual_net_MWh, CRF, LCOE)
