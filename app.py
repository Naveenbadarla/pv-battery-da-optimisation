import streamlit as st
import pandas as pd
import io
import math


# =====================================================================
#               CORE CALCULATION LOGIC (YEARLY, SINGLE SCENARIO)
# =====================================================================

def compute_yearly_scenario(
    load_kwh: float,
    pv_kwp: float,
    pv_yield: float,
    grid_price: float,
    fit_price: float,
    batt_capacity: float,
    batt_efficiency: float,
    cycles_per_day: float,
    sc_ratio_no_batt: float,
    da_spread: float,
    opt_capture: float,
    nonopt_capture: float,
) -> pd.DataFrame:
    """
    Yearly model with three configs:
      1) No battery
      2) Battery – non-optimised
      3) Battery – DA-optimised
    """
    pv_gen = pv_kwp * pv_yield

    # ----------------- No battery -----------------
    pv_direct_sc = min(load_kwh * sc_ratio_no_batt, pv_gen)
    pv_export_no_batt = max(0.0, pv_gen - pv_direct_sc)
    grid_import_no_batt = max(0.0, load_kwh - pv_direct_sc)

    cost_no_batt = grid_import_no_batt * grid_price
    revenue_no_batt = pv_export_no_batt * fit_price
    net_no_batt = cost_no_batt - revenue_no_batt

    # ----------------- Battery baseline -----------------
    batt_theoretical = batt_capacity * batt_efficiency * cycles_per_day * 365.0
    remaining_load = max(0.0, load_kwh - pv_direct_sc)
    batt_usable = min(batt_theoretical, remaining_load)  # usable kWh delivered to load

    pv_to_batt = batt_usable / batt_efficiency if batt_efficiency > 0 else 0.0
    pv_export_batt = max(0.0, pv_gen - pv_direct_sc - pv_to_batt)
    grid_import_batt = max(0.0, load_kwh - (pv_direct_sc + batt_usable))

    cost_batt_base = grid_import_batt * grid_price
    revenue_batt = pv_export_batt * fit_price
    net_batt_base = cost_batt_base - revenue_batt

    # Arbitrage energy: only if there was grid import in no-battery case
    arbitrage_energy = batt_usable if grid_import_no_batt > 0 else 0.0

    spread_non = da_spread * nonopt_capture
    spread_opt = da_spread * opt_capture

    arbitrage_non = arbitrage_energy * spread_non
    arbitrage_opt = arbitrage_energy * spread_opt

    net_batt_non = net_batt_base - arbitrage_non
    net_batt_opt = net_batt_base - arbitrage_opt

    df = pd.DataFrame(
        [
            {
                "Configuration": "No battery",
                "PV generation (kWh/yr)": pv_gen,
                "PV self-consumption (kWh/yr)": pv_direct_sc,
                "Battery → load (kWh/yr)": 0.0,
                "PV export (kWh/yr)": pv_export_no_batt,
                "Grid import (kWh/yr)": grid_import_no_batt,
                "Grid cost (€)": cost_no_batt,
                "EEG revenue (€)": revenue_no_batt,
                "DA arbitrage (€)": 0.0,
                "Net annual cost (€)": net_no_batt,
            },
            {
                "Configuration": "Battery – non-optimised",
                "PV generation (kWh/yr)": pv_gen,
                "PV self-consumption (kWh/yr)": pv_direct_sc,
                "Battery → load (kWh/yr)": batt_usable,
                "PV export (kWh/yr)": pv_export_batt,
                "Grid import (kWh/yr)": grid_import_batt,
                "Grid cost (€)": cost_batt_base,
                "EEG revenue (€)": revenue_batt,
                "DA arbitrage (€)": arbitrage_non,
                "Net annual cost (€)": net_batt_non,
            },
            {
                "Configuration": "Battery – DA-optimised",
                "PV generation (kWh/yr)": pv_gen,
                "PV self-consumption (kWh/yr)": pv_direct_sc,
                "Battery → load (kWh/yr)": batt_usable,
                "PV export (kWh/yr)": pv_export_batt,
                "Grid import (kWh/yr)": grid_import_batt,
                "Grid cost (€)": cost_batt_base,
                "EEG revenue (€)": revenue_batt,
                "DA arbitrage (€)": arbitrage_opt,
                "Net annual cost (€)": net_batt_opt,
            },
        ]
    )
    return df


# =====================================================================
#        SIMPLE HOURLY SIMULATION (1 TYPICAL DAY, DEMONSTRATION)
# =====================================================================

def synth_hourly_profiles(load_kwh: float, pv_kwp: float, pv_yield: float):
    """
    Create synthetic hourly arrays for one 'typical' day:
    - load_profile: shape(24)
    - pv_profile: shape(24)
    - price_profile: shape(24)
    """
    # Approx daily load
    daily_load = load_kwh / 365.0

    # Simple shape: more usage morning/evening
    base_load = []
    for h in range(24):
        if 0 <= h <= 5:
            base_load.append(0.5)
        elif 6 <= h <= 8:
            base_load.append(1.0)
        elif 9 <= h <= 16:
            base_load.append(0.7)
        elif 17 <= h <= 21:
            base_load.append(1.3)
        else:
            base_load.append(0.8)
    total_base = sum(base_load)
    load_profile = [daily_load * v / total_base for v in base_load]

    # Simple PV: bell between 8–18
    daily_pv = pv_kwp * pv_yield / 365.0
    pv_raw = []
    for h in range(24):
        if 8 <= h <= 18:
            x = (h - 8) / 10.0 * math.pi  # 0 → π
            pv_raw.append(max(0.0, math.sin(x)))
        else:
            pv_raw.append(0.0)
    total_raw = sum(pv_raw) or 1.0
    pv_profile = [daily_pv * v / total_raw for v in pv_raw]

    # Simple price: low at night, medium midday, high in evening
    price_profile = []
    for h in range(24):
        if 0 <= h <= 5:
            price_profile.append(0.20)
        elif 6 <= h <= 15:
            price_profile.append(0.26)
        elif 16 <= h <= 21:
            price_profile.append(0.38)
        else:
            price_profile.append(0.24)

    return load_profile, pv_profile, price_profile


def simulate_hourly_battery(load_profile, pv_profile, price_profile,
                            batt_capacity: float, batt_eff: float):
    """
    Simple hourly simulation for one day:
    - Battery charges from PV surplus first,
    - Then optionally from grid in cheap hours,
    - Discharges in expensive hours.
    Returns dataframe with SoC, flows, etc.
    """
    hours = list(range(24))
    soc = 0.0
    max_soc = batt_capacity
    rows = []

    cheap_threshold = sorted(price_profile)[7]   # roughly lower 1/3
    expensive_threshold = sorted(price_profile)[-7]  # roughly upper 1/3

    for h in hours:
        load = load_profile[h]
        pv = pv_profile[h]
        price = price_profile[h]

        soc_start = soc
        pv_to_load = min(load, pv)
        load_remaining = load - pv_to_load
        pv_surplus = pv - pv_to_load

        # Charge from PV surplus
        pv_to_batt = 0.0
        if pv_surplus > 0 and soc < max_soc:
            charge_possible = (max_soc - soc) / batt_eff
            charge = min(charge_possible, pv_surplus)
            pv_to_batt = charge
            soc += charge * batt_eff
            pv_surplus -= charge

        # Optionally charge from grid if cheap
        grid_to_batt = 0.0
        if price <= cheap_threshold and soc < max_soc:
            charge_possible = (max_soc - soc) / batt_eff
            grid_to_batt = min(charge_possible, max(0.0, 0.5 - pv_to_load))
            soc += grid_to_batt * batt_eff

        # Discharge in expensive hours to supply remaining load
        batt_to_load = 0.0
        if price >= expensive_threshold and soc > 0 and load_remaining > 0:
            discharge_possible = soc
            discharge = min(discharge_possible, load_remaining / batt_eff)
            batt_to_load = discharge * batt_eff
            soc -= discharge
            load_remaining -= batt_to_load

        # Remaining load from grid
        grid_to_load = max(0.0, load_remaining)

        rows.append(
            {
                "hour": h,
                "price (€/kWh)": price,
                "load (kWh)": load,
                "pv (kWh)": pv,
                "pv → load (kWh)": pv_to_load,
                "pv → batt (kWh)": pv_to_batt,
                "grid → batt (kWh)": grid_to_batt,
                "batt → load (kWh)": batt_to_load,
                "grid → load (kWh)": grid_to_load,
                "SoC (kWh)": soc,
                "SoC start (kWh)": soc_start,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
#                      SIMPLE RECOMMENDATION ENGINE
# =====================================================================

def generate_recommendation(df_yearly: pd.DataFrame) -> str:
    costs = df_yearly.set_index("Configuration")["Net annual cost (€)"]
    nb = float(costs["No battery"])
    bn = float(costs["Battery – non-optimised"])
    bo = float(costs["Battery – DA-optimised"])

    savings_batt = nb - bn
    extra_opt = bn - bo

    msg_lines = []

    # Battery value
    if savings_batt > 0:
        msg_lines.append(
            f"- ✅ The battery reduces your annual net cost by **≈ {savings_batt:,.0f} €** compared to having no battery."
        )
    else:
        msg_lines.append(
            "- ⚠️ In this configuration the battery does **not** reduce your net cost. "
            "Try changing load, PV size or prices."
        )

    # Optimisation value
    if extra_opt > 0:
        msg_lines.append(
            f"- 🧠 Smart day-ahead optimisation adds **≈ {extra_opt:,.0f} €** extra savings on top of the battery."
        )
    else:
        msg_lines.append(
            "- ℹ️ Extra savings from optimisation are very small with these assumptions."
        )

    # Sign of costs
    if bo < 0:
        msg_lines.append(
            "- 💶 With a DA-optimised battery, you become a **net earner**: EEG income is higher than your grid costs."
        )
    elif bn < 0:
        msg_lines.append(
            "- 💶 With a simple battery, you are already a **net earner** over the year."
        )
    elif nb < 0:
        msg_lines.append(
            "- 💶 Even without a battery, your PV exports make you a net earner."
        )
    else:
        msg_lines.append(
            "- 💡 All configurations still have a positive net cost, but the battery and optimisation reduce it."
        )

    return "\n".join(msg_lines)


# =====================================================================
#                            STREAMLIT APP
# =====================================================================

def main():
    st.set_page_config(page_title="PV + Battery + DA Optimisation", layout="wide")

    st.title("⚡ PV + Battery + Day-Ahead Optimisation (Germany / EEG)")

    st.markdown(
        """
This app shows how much you **pay or earn per year** with:
- a PV system (with EEG feed-in),
- an optional **battery** (only supplies the house, no battery → grid export),
- and **simple vs smart (DA-optimised)** battery control.

Use the sidebar to change assumptions – the results update instantly.
"""
    )

    # ---------------------- Sidebar inputs -----------------------
    st.sidebar.header("🔧 System setup")

    with st.sidebar.expander("💡 Quick explanation", expanded=True):
        st.markdown(
            """
### Energy flow

```text
        ☀️ PV
          │
          ├──→ 🏠 Home (direct use)
          │
          ├──→ 🔋 Battery → 🏠 Home (later use)
          │
          └──→ 🔌 Grid (EEG export)
Net cost = Grid cost − EEG revenue
  # Top-level metrics
  costs = df_yearly.set_index("Configuration")["Net annual cost (€)"]
  nb = float(costs["No battery"])
  bn = float(costs["Battery – non-optimised"])
  bo = float(costs["Battery – DA-optimised"])

  savings_batt = nb - bn
  extra_opt = bn - bo

  col1, col2, col3 = st.columns(3)
  col1.metric("Net annual cost – No battery", f"{nb:,.0f} €")
  col2.metric(
      "Net annual cost – Battery (simple)",
      f"{bn:,.0f} €",
      f"{savings_batt:,.0f} € vs no battery",
  )
  col3.metric(
      "Net annual cost – Battery (optimised)",
      f"{bo:,.0f} €",
      f"{extra_opt:,.0f} € extra vs simple",
  )

  st.subheader("Detailed yearly energy & cost table")
  st.dataframe(df_display, use_container_width=True)

  # Bar chart
  st.subheader("Net annual cost by configuration (lower is better)")
  st.bar_chart(df_yearly.set_index("Configuration")["Net annual cost (€)"])

  # CSV download
  csv_buf = io.StringIO()
  df_display.to_csv(csv_buf, index=False)
  st.download_button(
      "⬇️ Download results as CSV",
      data=csv_buf.getvalue(),
      file_name="pv_battery_results.csv",
      mime="text/csv",
  )

  # Short explanation
  with st.expander("📘 Short explanation of results", expanded=False):
      st.markdown(
          """
  # Recommendation engine
  st.subheader("🧠 Plain-language interpretation")
  st.markdown(generate_recommendation(df_yearly))
  st.markdown(
      """
# -----------------------------------------------------------------
# TAB: OPTIMISATION LOGIC
# -----------------------------------------------------------------
with tab_logic:
    st.header("⚙️ Optimisation logic")

    st.markdown(
        """
Extra value = shifted_energy '×' (optimised_spread − nonoptimised_spread)
    st.subheader("Graphical energy flow (conceptual)")
    st.graphviz_chart(
        """
PV [label="☀️ PV"];
HOME [label="🏠 Home"];
BATT [label="🔋 Battery"];
GRID [label="🔌 Grid (EEG)"];

PV -> HOME [label="direct\\nself-consumption"];
PV -> BATT [label="charge"];
BATT -> HOME [label="discharge"];
PV -> GRID [label="export\\n(EEG revenue)"];
GRID -> HOME [label="import\\n(grid cost)"];
# -----------------------------------------------------------------
# TAB: HOW TO READ RESULTS (DETAILED)
# -----------------------------------------------------------------
with tab_read:
    st.header("🧭 How to read the results (detailed)")

    st.markdown(
        """
Net cost = (grid cost) − (EEG revenue)
Savings_from_battery = net_cost(no_batt) − net_cost(battery_simple)
Extra_from_optimisation = net_cost(battery_simple) − net_cost(battery_optimised)
  st.markdown(
      """
    if batt_capacity <= 0:
        st.info(
            "Set a non-zero battery capacity in the sidebar to see the SoC simulation."
        )
    else:
        load_prof, pv_prof, price_prof = synth_hourly_profiles(
            load_kwh, pv_kwp, pv_yield
        )
        df_hourly = simulate_hourly_battery(
            load_prof, pv_prof, price_prof, batt_capacity, batt_eff
        )

        st.subheader("Hourly load and PV")
        st.line_chart(
            df_hourly.set_index("hour")[["load (kWh)", "pv (kWh)"]]
        )

        st.subheader("Battery state of charge (SoC)")
        st.line_chart(df_hourly.set_index("hour")[["SoC (kWh)"]])

        st.subheader("Grid vs battery vs PV to load")
        st.area_chart(
            df_hourly.set_index("hour")[
                ["pv → load (kWh)", "batt → load (kWh)", "grid → load (kWh)"]
            ]
        )

        with st.expander("See hourly data table"):
            st.dataframe(df_hourly.round(3), use_container_width=True)

# -----------------------------------------------------------------
# TAB: SAVED SCENARIOS (IN-SESSION)
# -----------------------------------------------------------------
with tab_save:
    st.header("💾 Save scenarios (this session only)")

    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = []

    scenario_name = st.text_input(
        "Name this scenario",
        value=f"Scenario: load={int(load_kwh)} kWh, PV={pv_kwp} kWp",
    )

    if st.button("💾 Save current scenario"):
        st.session_state.saved_scenarios.append(
            {
                "name": scenario_name,
                "results": df_display.to_dict(orient="records"),
            }
        )
        st.success("Scenario saved (only for this browser session).")

    if st.session_state.saved_scenarios:
        st.subheader("Saved scenarios in this session")
        for i, s in enumerate(st.session_state.saved_scenarios):
            st.markdown(f"### {i+1}. {s['name']}")
            df_s = pd.DataFrame(s["results"])
            st.dataframe(df_s, use_container_width=True)

        # Allow CSV download of all saved scenarios metadata
        flat_rows = []
        for s in st.session_state.saved_scenarios:
            for row in s["results"]:
                flat = {"Scenario name": s["name"]}
                flat.update(row)
                flat_rows.append(flat)
        df_flat = pd.DataFrame(flat_rows)
        buf_all = io.StringIO()
        df_flat.to_csv(buf_all, index=False)
        st.download_button(
            "⬇️ Download all saved scenarios (CSV)",
            data=buf_all.getvalue(),
            file_name="saved_scenarios.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No scenarios saved yet. Configure a setup and click 'Save current scenario'."
        )
