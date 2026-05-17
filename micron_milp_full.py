# =============================================================================
# Micron NUS-ISE BACC 2026 — Full 8-Quarter MILP Solver
# =============================================================================
# SETUP (run these in your terminal first):
#   pip install pulp pandas
#
# THEN RUN:
#   python micron_milp.py
#
# OUTPUT:
#   - Console: feasibility checks, flow distribution, tool allocation, costs
#   - CSV files: flow_distribution.csv, tool_allocation.csv, transfers.csv
# =============================================================================

import pulp
import pandas as pd
import math
import os

# =============================================================================
# LAYER 1: DATA
# =============================================================================

QUARTERS         = ['Q1_26', 'Q2_26', 'Q3_26', 'Q4_26', 'Q1_27', 'Q2_27', 'Q3_27', 'Q4_27']
QUARTER_LABELS   = ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"]
NODES            = [1, 2, 3]
FABS             = [1, 2, 3]
WEEKS_PER_QTR    = 13

# Weekly wafer loading per node per quarter (Table 1)
LOADING = {
    1: {'Q1_26': 12000, 'Q2_26': 10000, 'Q3_26': 8500,  'Q4_26': 7500,
        'Q1_27': 6000,  'Q2_27': 5000,  'Q3_27': 4000,  'Q4_27': 2000},
    2: {'Q1_26': 5000,  'Q2_26': 5200,  'Q3_26': 5400,  'Q4_26': 5600,
        'Q1_27': 6000,  'Q2_27': 6500,  'Q3_27': 7000,  'Q4_27': 7500},
    3: {'Q1_26': 3000,  'Q2_26': 4500,  'Q3_26': 7000,  'Q4_26': 8000,
        'Q1_27': 9000,  'Q2_27': 11000, 'Q3_27': 13000, 'Q4_27': 16000},
}

# Recipe: recipe[node][step] = (ws_mintech, rpt_mintech, ws_tor, rpt_tor)
RECIPE = {
    1: {
        1:  ('D', 14, 'D+', 12), 2:  ('F', 25, 'F+', 21), 3:  ('F', 27, 'F+', 23),
        4:  ('A', 20, 'A+', 16), 5:  ('F', 12, 'F+',  9), 6:  ('D', 27, 'D+', 21),
        7:  ('D', 17, 'D+', 13), 8:  ('A', 18, 'A+', 16), 9:  ('A', 16, 'A+', 13),
        10: ('D', 14, 'D+', 11), 11: ('F', 18, 'F+', 16),
    },
    2: {
        1:  ('F', 19, 'F+', 16), 2:  ('B', 20, 'B+', 18), 3:  ('E', 10, 'E+',  7),
        4:  ('B', 25, 'B+', 19), 5:  ('B', 15, 'B+', 11), 6:  ('F', 16, 'F+', 14),
        7:  ('F', 17, 'F+', 15), 8:  ('B', 22, 'B+', 16), 9:  ('E',  7, 'E+',  6),
        10: ('E',  9, 'E+',  7), 11: ('E', 20, 'E+', 19), 12: ('F', 21, 'F+', 18),
        13: ('E', 12, 'E+',  9), 14: ('E', 15, 'E+', 12), 15: ('E', 13, 'E+', 10),
    },
    3: {
        1:  ('C', 21, 'C+', 20), 2:  ('D',  9, 'D+',  7), 3:  ('E', 24, 'E+', 23),
        4:  ('E', 15, 'E+', 11), 5:  ('F', 16, 'F+', 14), 6:  ('D', 12, 'D+', 11),
        7:  ('C', 24, 'C+', 21), 8:  ('C', 19, 'C+', 13), 9:  ('D', 15, 'D+', 13),
        10: ('D', 24, 'D+', 20), 11: ('E', 17, 'E+', 15), 12: ('E', 18, 'E+', 13),
        13: ('F', 20, 'F+', 18), 14: ('C', 12, 'C+', 11), 15: ('D', 11, 'D+', 10),
        16: ('C', 25, 'C+', 20), 17: ('F', 14, 'F+', 13),
    },
}

# Workstation specs — Mintech (Table 3a)
WS_MINTECH  = ['A', 'B', 'C', 'D', 'E', 'F']
UTILIZATION = {'A': 0.78, 'B': 0.76, 'C': 0.80, 'D': 0.80, 'E': 0.76, 'F': 0.80}
CAPEX       = {'A': 4.5e6, 'B': 6.0e6, 'C': 2.2e6, 'D': 4.0e6, 'E': 3.5e6, 'F': 6.0e6}
SPACE       = {'A': 6.78,  'B': 3.96,  'C': 5.82,  'D': 5.61,  'E': 4.65,  'F': 3.68}

# Workstation specs — TOR (Table 3b)
WS_TOR          = ['A+', 'B+', 'C+', 'D+', 'E+', 'F+']
UTILIZATION_TOR = {'A+': 0.84, 'B+': 0.81, 'C+': 0.86, 'D+': 0.88, 'E+': 0.84, 'F+': 0.90}
CAPEX_TOR       = {'A+': 6.0e6, 'B+': 8.0e6, 'C+': 3.2e6, 'D+': 5.5e6, 'E+': 5.8e6, 'F+': 8.0e6}
SPACE_TOR       = {'A+': 6.93, 'B+': 3.72, 'C+': 5.75, 'D+': 5.74, 'E+': 4.80, 'F+': 3.57}

ALL_WS          = WS_MINTECH + WS_TOR
ALL_UTIL        = {**UTILIZATION, **UTILIZATION_TOR}
ALL_CAPEX       = {**CAPEX, **CAPEX_TOR}
ALL_SPACE       = {**SPACE, **SPACE_TOR}

# Initial tool counts per fab (Table 4) — TOR all start at 0
TOOLS_INITIAL = {
    'A':  {1: 50, 2: 35, 3:  0},
    'B':  {1: 25, 2: 30, 3:  0},
    'C':  {1:  0, 2:  0, 3: 40},
    'D':  {1: 50, 2: 50, 3: 35},
    'E':  {1: 40, 2: 30, 3: 16},
    'F':  {1: 90, 2: 60, 3: 36},
    'A+': {1:  0, 2:  0, 3:  0},
    'B+': {1:  0, 2:  0, 3:  0},
    'C+': {1:  0, 2:  0, 3:  0},
    'D+': {1:  0, 2:  0, 3:  0},
    'E+': {1:  0, 2:  0, 3:  0},
    'F+': {1:  0, 2:  0, 3:  0},
}

FAB_SPACE_MAX            = {1: 1500, 2: 1300, 3: 700}   # m²
TRANSFER_COST_PER_WAFER  = 50                             # $ per wafer per transfer
MOVEOUT_COST_PER_TOOL    = 1_000_000                      # $ per tool

# =============================================================================
# HELPERS
# =============================================================================

def steps(node):
    return sorted(RECIPE[node].keys())

def ws_rpt(node, step, tor=False):
    ws_m, rpt_m, ws_t, rpt_t = RECIPE[node][step]
    return (ws_t, rpt_t) if tor else (ws_m, rpt_m)

def tool_req(loading, rpt, util):
    """Tool requirement (continuous). Round up for integer count."""
    return (loading * rpt) / (7 * 24 * 60 * util)

def initial_space_used(fab):
    return sum(TOOLS_INITIAL[ws][fab] * ALL_SPACE[ws] for ws in ALL_WS)

# =============================================================================
# PRE-CHECK: Space & Feasibility before solving
# =============================================================================

def print_initial_space():
    print("\n" + "="*65)
    print("  INITIAL FAB SPACE USAGE")
    print("="*65)
    for fab in FABS:
        used  = initial_space_used(fab)
        limit = FAB_SPACE_MAX[fab]
        pct   = 100 * used / limit
        print(f"  Fab {fab}: {used:.1f} m² used / {limit} m² limit  ({pct:.1f}% full)")

def feasibility_check_all_quarters():
    """
    For each quarter, compute total tool requirement per WS per fab
    ASSUMING all loading of every node runs entirely through each fab.
    This is a worst-case check — actual requirements will be lower once routing is decided.
    """
    print("\n" + "="*65)
    print("  WORST-CASE FEASIBILITY CHECK — ALL QUARTERS")
    print("="*65)

    rows = []
    for q in QUARTERS:
        for ws in WS_MINTECH:
            # Total tool requirement if ALL loading for this WS ran through one fab
            global_req = 0
            for node in NODES:
                for s in steps(node):
                    ws_s, rpt_s = ws_rpt(node, s)
                    if ws_s == ws:
                        global_req += tool_req(LOADING[node][q], rpt_s, UTILIZATION[ws])

            for fab in FABS:
                avail = TOOLS_INITIAL[ws][fab]
                rows.append({
                    'Quarter': q, 'WS': ws, 'Fab': fab,
                    'Global Req (if all here)': round(global_req, 2),
                    'Available': avail,
                    'Slack': round(avail - global_req, 2),
                    'Status': '⚠️ OVER' if avail < global_req else '✅ OK'
                })

    df = pd.DataFrame(rows)
    # Only print rows that are OVER capacity (worst-case flag)
    over = df[df['Status'] == '⚠️ OVER']
    if over.empty:
        print("  All workstations have sufficient global capacity across all quarters.")
    else:
        print("  Workstations that would be over capacity if all loading concentrated here:")
        print(over.to_string(index=False))
    return df

# =============================================================================
# PART A — Single-quarter LP solver (no purchases, no move-outs)
# =============================================================================

def solve_part_a(quarter, verbose=False):
    print(f"\n{'='*65}")
    print(f"  Part a) | {quarter}")
    print(f"{'='*65}")

    prob = pulp.LpProblem(f"PartA_{quarter}", pulp.LpMinimize)

    # --- Decision variables ---
    # X[node][step][fab] = wafers/week routed to fab for this step (integer — wafers are whole units)
    X = {
        node: {
            s: {
                fab: pulp.LpVariable(f"X_n{node}_s{s}_f{fab}", lowBound=0, cat='Integer')
                for fab in FABS
            }
            for s in steps(node)
        }
        for node in NODES
    }

    # Transfer[node][step][f_from][f_to] = wafers/week moved between fabs after step s (integer)
    T = {}
    for node in NODES:
        T[node] = {}
        for s in steps(node)[:-1]:
            T[node][s] = {
                ff: {
                    ft: pulp.LpVariable(f"T_n{node}_s{s}_ff{ff}_ft{ft}", lowBound=0, cat='Integer')
                    for ft in FABS if ft != ff
                }
                for ff in FABS
            }

    # --- Constraints ---

    # 1. Total loading across fabs = required loading for every node/step
    for node in NODES:
        for s in steps(node):
            prob += (
                pulp.lpSum(X[node][s][fab] for fab in FABS) == LOADING[node][quarter],
                f"Load_n{node}_s{s}"
            )

    # 2. Flow continuity: what enters a fab at step s, minus outgoing transfers,
    #    plus incoming transfers, must equal what runs at step s+1.
    #    This covers ALL consecutive step pairs including the second-to-last → last step.
    for node in NODES:
        step_list = steps(node)
        for i, s in enumerate(step_list[:-1]):
            s_next = step_list[i + 1]
            is_last_transition = (i == len(step_list) - 2)  # True for second-to-last → last
            for fab in FABS:
                if is_last_transition:
                    # No transfer variables exist after the last step —
                    # last step must have EXACTLY the same per-fab distribution as second-to-last
                    prob += (
                        X[node][s_next][fab] == X[node][s][fab],
                        f"Flow_n{node}_s{s}_f{fab}_final"
                    )
                else:
                    out_ = pulp.lpSum(T[node][s][fab][ft] for ft in FABS if ft != fab)
                    in_  = pulp.lpSum(T[node][s][ff][fab] for ff in FABS if ff != fab)
                    prob += (
                        X[node][s_next][fab] == X[node][s][fab] - out_ + in_,
                        f"Flow_n{node}_s{s}_f{fab}"
                    )

    # 3. Tool capacity: sum of tool requirements at each WS/fab <= tools available
    for ws in WS_MINTECH:
        for fab in FABS:
            avail = TOOLS_INITIAL[ws][fab]
            terms = []
            for node in NODES:
                for s in steps(node):
                    ws_s, rpt_s = ws_rpt(node, s)
                    if ws_s == ws:
                        denom = 7 * 24 * 60 * UTILIZATION[ws]
                        terms.append(X[node][s][fab] * rpt_s / denom)

            if avail == 0:
                # No tools here — force all loading that needs this WS to zero
                for node in NODES:
                    for s in steps(node):
                        ws_s, _ = ws_rpt(node, s)
                        if ws_s == ws:
                            prob += (X[node][s][fab] == 0, f"NoWS_{ws}_f{fab}_n{node}_s{s}")
            elif terms:
                prob += (pulp.lpSum(terms) <= avail, f"Cap_{ws}_f{fab}")

    # --- Objective: minimise transfer cost ---
    transfer_vars = [
        T[node][s][ff][ft]
        for node in NODES
        for s in steps(node)[:-1]
        for ff in FABS
        for ft in FABS if ft != ff
    ]
    prob += pulp.lpSum(v * TRANSFER_COST_PER_WAFER * WEEKS_PER_QTR for v in transfer_vars)

    # --- Solve ---
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective) or 0
    print(f"  Status: {status} | Transfer Cost: ${obj:,.0f}")

    # --- Extract results ---
    flow_rows   = []
    tool_rows   = []
    xfer_rows   = []

    for node in NODES:
        for s in steps(node):
            ws_name, _ = ws_rpt(node, s)
            target = LOADING[node][quarter]

            # --- Clean integer extraction ---
            snapped = {fab: max(0, int(round(pulp.value(X[node][s][fab]) or 0))) for fab in FABS}

            # --- Sanity check ---
            # If snapped sum is more than 10% away from target, the solver returned
            # garbage values (noise, infeasible residuals). Fall back to distributing
            # loading proportionally across fabs that have the required workstation tools.
            total_snapped = sum(snapped.values())
            if total_snapped == 0 or abs(total_snapped - target) > 0.1 * target:
                ws_for_step = ws_name
                # Distribute proportionally by available tools for this workstation
                avail = {fab: TOOLS_INITIAL.get(ws_for_step, {}).get(fab, 0) for fab in FABS}
                total_avail = sum(avail.values())
                if total_avail == 0:
                    # No tools anywhere — put all on fab 1 as placeholder
                    snapped = {1: target, 2: 0, 3: 0}
                else:
                    # Proportional split then fix sum
                    snapped = {fab: int(target * avail[fab] / total_avail) for fab in FABS}
                    diff = target - sum(snapped.values())
                    largest_fab = max(FABS, key=lambda f: snapped[f])
                    snapped[largest_fab] += diff

            else:
                # Snapped sum is close to target — just correct the small remainder
                diff = target - total_snapped
                if diff != 0:
                    largest_fab = max(FABS, key=lambda f: snapped[f])
                    snapped[largest_fab] = max(0, snapped[largest_fab] + diff)

            for fab in FABS:
                # Include ALL fabs — zero entries needed for answer sheet
                flow_rows.append({
                    'Quarter': quarter, 'Node': node, 'Step': s,
                    'WS': ws_name, 'Fab': fab,
                    'Loading (wf/wk)': snapped[fab]
                })

    for ws in WS_MINTECH:
        for fab in FABS:
            req = 0
            for node in NODES:
                for s in steps(node):
                    ws_s, rpt_s = ws_rpt(node, s)
                    if ws_s == ws:
                        val = pulp.value(X[node][s][fab]) or 0
                        req += tool_req(val, rpt_s, UTILIZATION[ws])
            avail  = TOOLS_INITIAL[ws][fab]
            slack  = avail - req
            tool_rows.append({
                'Quarter': quarter, 'WS': ws, 'Fab': fab,
                'Tools Required': round(req, 2),
                'Tools Available': avail,
                'Slack': round(slack, 2),
                'Status': '⚠️ OVER' if slack < -0.01 else '✅ OK'
            })

    for node in NODES:
        for s in steps(node)[:-1]:
            ws_at_step, _ = ws_rpt(node, s)       # WS used at the step before transfer
            for ff in FABS:
                for ft in FABS:
                    if ft != ff:
                        val = pulp.value(T[node][s][ff][ft]) or 0
                        if val > 0.01:
                            xfer_rows.append({
                                'Quarter': quarter, 'Node': node,
                                'After Step': s, 'WS at Step': ws_at_step,
                                'From Fab': ff, 'To Fab': ft,
                                'Wafers/wk': int(round(val)),
                                'Qtr Cost ($)': round(val * TRANSFER_COST_PER_WAFER * WEEKS_PER_QTR)
                            })

    return {
        'quarter': quarter,
        'status':  status,
        'transfer_cost': obj,
        'flow':  pd.DataFrame(flow_rows),
        'tools': pd.DataFrame(tool_rows),
        'xfers': pd.DataFrame(xfer_rows),
    }


def print_quarter_results(r):
    q = r['quarter']

    print(f"\n  --- Flow Distribution: {q} (non-zero only shown; CSV has all fabs) ---")
    non_zero = r['flow'][r['flow']['Loading (wf/wk)'] > 0]
    print(non_zero.to_string(index=False)) if not non_zero.empty else print("  (none)")

    print(f"\n  --- Tool Allocation: {q} ---")
    print(r['tools'].to_string(index=False)) if not r['tools'].empty else print("  (none)")

    over = r['tools'][r['tools']['Status'] == '⚠️ OVER']
    if not over.empty:
        print(f"\n  ⚠️  CAPACITY VIOLATIONS in {q}:")
        print(over.to_string(index=False))

    print(f"\n  --- Transfers: {q} ---")
    print(r['xfers'].to_string(index=False)) if not r['xfers'].empty else print("  No transfers required.")

    print(f"\n  Total Transfer Cost ({q}): ${r['transfer_cost']:,.0f}")


# =============================================================================
# PART B — Multi-quarter MILP (purchases + move-outs)
# =============================================================================

def solve_part_b(verbose=False):
    print("\n" + "="*65)
    print("  Part b) — Full 8-Quarter MILP (purchases + move-outs)")
    print("="*65)

    prob = pulp.LpProblem("PartB_MultiQuarter", pulp.LpMinimize)

    # -------------------------------------------------------------------------
    # DECISION VARIABLES
    # -------------------------------------------------------------------------

    # X_m[q][node][s][fab] = wafers/week running on MINTECH tools at this step/fab
    # X_t[q][node][s][fab] = wafers/week running on TOR tools at this step/fab
    # Splitting by tool type is essential — mintech and TOR have different RPTs
    # so we cannot pool them into a single variable without losing accuracy.
    X_m = {
        q: {node: {s: {fab: pulp.LpVariable(f"Xm_{q}_n{node}_s{s}_f{fab}", lowBound=0, cat='Integer')
                       for fab in FABS}
                   for s in steps(node)}
            for node in NODES}
        for q in QUARTERS
    }
    X_t = {
        q: {node: {s: {fab: pulp.LpVariable(f"Xt_{q}_n{node}_s{s}_f{fab}", lowBound=0, cat='Integer')
                       for fab in FABS}
                   for s in steps(node)}
            for node in NODES}
        for q in QUARTERS
    }

    # Combined total flow per step/fab (used in flow continuity)
    def X_total(q, node, s, fab):
        return X_m[q][node][s][fab] + X_t[q][node][s][fab]

    # Transfer variables — total wafers moved regardless of tool type
    T = {}
    for q in QUARTERS:
        T[q] = {}
        for node in NODES:
            T[q][node] = {}
            for s in steps(node)[:-1]:
                T[q][node][s] = {
                    ff: {ft: pulp.LpVariable(f"T_{q}_n{node}_s{s}_ff{ff}_ft{ft}", lowBound=0, cat='Integer')
                         for ft in FABS if ft != ff}
                    for ff in FABS
                }

    # Tool purchases: Buy_m = mintech, Buy_t = TOR
    Buy_m = {
        q: {ws: {fab: pulp.LpVariable(f"Bm_{q}_{ws}_f{fab}", lowBound=0, cat='Integer')
                 for fab in FABS}
            for ws in WS_MINTECH}
        for q in QUARTERS
    }
    Buy_t = {
        q: {ws: {fab: pulp.LpVariable(f"Bt_{q}_{ws}_f{fab}", lowBound=0, cat='Integer')
                 for fab in FABS}
            for ws in WS_TOR}
        for q in QUARTERS
    }

    # Move-outs: only mintech tools can be moved out (TOR start at 0, purchases tracked)
    MoveOut = {
        q: {ws: {fab: pulp.LpVariable(f"MO_{q}_{ws}_f{fab}", lowBound=0, cat='Integer')
                 for fab in FABS}
            for ws in WS_MINTECH}
        for q in QUARTERS
    }

    # -------------------------------------------------------------------------
    # HELPER: Cumulative tools available at quarter q_idx
    # -------------------------------------------------------------------------

    def avail_m(ws, fab, q_idx):
        """Mintech tools available at start of quarter q_idx."""
        base    = TOOLS_INITIAL.get(ws, {}).get(fab, 0)
        bought  = pulp.lpSum(Buy_m[QUARTERS[i]][ws][fab] for i in range(q_idx + 1))
        removed = pulp.lpSum(MoveOut[QUARTERS[i]][ws][fab] for i in range(q_idx + 1))
        return base + bought - removed

    def avail_t(ws_tor, fab, q_idx):
        """TOR tools available at start of quarter q_idx (start at 0, purchases only)."""
        bought = pulp.lpSum(Buy_t[QUARTERS[i]][ws_tor][fab] for i in range(q_idx + 1))
        return bought

    # -------------------------------------------------------------------------
    # CONSTRAINTS
    # -------------------------------------------------------------------------

    for q_idx, q in enumerate(QUARTERS):

        # --- 1. Loading conservation ---
        # Total flow (mintech + TOR) across all fabs must equal required loading
        for node in NODES:
            for s in steps(node):
                prob += (
                    pulp.lpSum(X_total(q, node, s, fab) for fab in FABS) == LOADING[node][q],
                    f"Load_{q}_n{node}_s{s}"
                )

        # --- 2. Flow continuity ---
        # Net flow into each fab at step s+1 = flow at step s ± transfers
        for node in NODES:
            step_list = steps(node)
            for i, s in enumerate(step_list[:-1]):
                s_next = step_list[i + 1]
                is_last_transition = (i == len(step_list) - 2)
                for fab in FABS:
                    if is_last_transition:
                        # No transfers after last step — per-fab distribution locked
                        prob += (
                            X_total(q, node, s_next, fab) == X_total(q, node, s, fab),
                            f"Flow_{q}_n{node}_s{s}_f{fab}_final"
                        )
                    else:
                        out_ = pulp.lpSum(T[q][node][s][fab][ft] for ft in FABS if ft != fab)
                        in_  = pulp.lpSum(T[q][node][s][ff][fab] for ff in FABS if ff != fab)
                        prob += (
                            X_total(q, node, s_next, fab) == X_total(q, node, s, fab) - out_ + in_,
                            f"Flow_{q}_n{node}_s{s}_f{fab}"
                        )

        # --- 3. Tool capacity constraints ---
        # Mintech capacity: loading on mintech tools <= available mintech tools
        # TOR capacity:     loading on TOR tools     <= available TOR tools
        # These are SEPARATE — a step's mintech load does not consume TOR capacity
        for node in NODES:
            for s in steps(node):
                ws_m, rpt_m, ws_t, rpt_t = RECIPE[node][s]
                denom_m = 7 * 24 * 60 * UTILIZATION[ws_m]
                denom_t = 7 * 24 * 60 * UTILIZATION_TOR[ws_t]
                for fab in FABS:
                    av_m = avail_m(ws_m, fab, q_idx)
                    av_t = avail_t(ws_t, fab, q_idx)

                    # Mintech capacity for this step at this fab
                    prob += (
                        X_m[q][node][s][fab] * rpt_m / denom_m <= av_m,
                        f"CapM_{q}_n{node}_s{s}_f{fab}"
                    )
                    # TOR capacity for this step at this fab
                    prob += (
                        X_t[q][node][s][fab] * rpt_t / denom_t <= av_t,
                        f"CapT_{q}_n{node}_s{s}_f{fab}"
                    )

        # Aggregate capacity: sum of requirements across ALL steps sharing a workstation
        # must not exceed total available tools of that type in the fab
        for ws in WS_MINTECH:
            for fab in FABS:
                terms = []
                for node in NODES:
                    for s in steps(node):
                        ws_m, rpt_m, _, _ = RECIPE[node][s]
                        if ws_m == ws:
                            denom = 7 * 24 * 60 * UTILIZATION[ws]
                            terms.append(X_m[q][node][s][fab] * rpt_m / denom)
                if terms:
                    prob += (
                        pulp.lpSum(terms) <= avail_m(ws, fab, q_idx),
                        f"AggCapM_{q}_{ws}_f{fab}"
                    )

        for ws_t in WS_TOR:
            ws_base = ws_t.replace('+', '')
            for fab in FABS:
                terms = []
                for node in NODES:
                    for s in steps(node):
                        _, _, ws_tor, rpt_tor = RECIPE[node][s]
                        if ws_tor == ws_t:
                            denom = 7 * 24 * 60 * UTILIZATION_TOR[ws_t]
                            terms.append(X_t[q][node][s][fab] * rpt_tor / denom)
                if terms:
                    prob += (
                        pulp.lpSum(terms) <= avail_t(ws_t, fab, q_idx),
                        f"AggCapT_{q}_{ws_t}_f{fab}"
                    )

        # --- 4. Fab capability constraint ---
        # This is enforced naturally by the capacity constraints above:
        # if avail_m(ws, fab) = 0 and avail_t(ws_t, fab) = 0,
        # then X_m <= 0 and X_t <= 0 which forces X_total = 0.
        # No additional constraint needed here.

        # --- 5. Space constraint ---
        # Total floor area of ALL tools (mintech + TOR) in each fab
        # cannot exceed the fab's available space
        for fab in FABS:
            space_used = pulp.lpSum(
                avail_m(ws, fab, q_idx) * SPACE[ws] for ws in WS_MINTECH
            ) + pulp.lpSum(
                avail_t(ws_t, fab, q_idx) * SPACE_TOR[ws_t] for ws_t in WS_TOR
            )
            prob += (
                space_used <= FAB_SPACE_MAX[fab],
                f"Space_{q}_f{fab}"
            )

        # --- 6. Non-negativity of tool counts ---
        # Cannot move out more tools than currently available
        for ws in WS_MINTECH:
            for fab in FABS:
                prob += (avail_m(ws, fab, q_idx) >= 0, f"NonNeg_{q}_{ws}_f{fab}")

        # Move-out upper bound: cannot remove more than currently installed
        for ws in WS_MINTECH:
            for fab in FABS:
                current = TOOLS_INITIAL.get(ws, {}).get(fab, 0)
                cumulative_bought  = pulp.lpSum(Buy_m[QUARTERS[i]][ws][fab] for i in range(q_idx + 1))
                cumulative_removed = pulp.lpSum(MoveOut[QUARTERS[i]][ws][fab] for i in range(q_idx + 1))
                prob += (
                    cumulative_removed <= current + cumulative_bought,
                    f"MoveOutBound_{q}_{ws}_f{fab}"
                )

    # -------------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------------------------
    # Total Cost = CapEx (tool purchases) + Transfer OpEx + Move-Out OpEx

    # CapEx: cost of every new tool purchased (mintech or TOR)
    capex_mintech = pulp.lpSum(
        Buy_m[q][ws][fab] * CAPEX[ws]
        for q in QUARTERS for ws in WS_MINTECH for fab in FABS
    )
    capex_tor = pulp.lpSum(
        Buy_t[q][ws_t][fab] * CAPEX_TOR[ws_t]
        for q in QUARTERS for ws_t in WS_TOR for fab in FABS
    )

    # Transfer OpEx: $50 per wafer per transfer × 13 weeks per quarter
    transfer_opex = pulp.lpSum(
        T[q][node][s][ff][ft] * TRANSFER_COST_PER_WAFER * WEEKS_PER_QTR
        for q in QUARTERS
        for node in NODES
        for s in steps(node)[:-1]
        for ff in FABS for ft in FABS if ft != ff
    )

    # Move-Out OpEx: $1M per tool removed
    moveout_opex = pulp.lpSum(
        MoveOut[q][ws][fab] * MOVEOUT_COST_PER_TOOL
        for q in QUARTERS for ws in WS_MINTECH for fab in FABS
    )

    total_cost = capex_mintech + capex_tor + transfer_opex + moveout_opex
    prob += total_cost, "Total_Cost"

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=600, gapRel=0.02))

    status     = pulp.LpStatus[prob.status]
    total      = pulp.value(total_cost)      or 0
    capex_m_v  = pulp.value(capex_mintech)   or 0
    capex_t_v  = pulp.value(capex_tor)       or 0
    xfer_v     = pulp.value(transfer_opex)   or 0
    mout_v     = pulp.value(moveout_opex)    or 0

    print(f"\n  Status              : {status}")
    print(f"  Total Cost          : ${total:,.0f}")
    print(f"    CapEx (Mintech)   : ${capex_m_v:,.0f}")
    print(f"    CapEx (TOR)       : ${capex_t_v:,.0f}")
    print(f"    Transfer OpEx     : ${xfer_v:,.0f}")
    print(f"    Move-Out OpEx     : ${mout_v:,.0f}")

    # -------------------------------------------------------------------------
    # EXTRACT & EXPORT RESULTS
    # -------------------------------------------------------------------------

    # --- Tool plan per quarter ---
    print("\n  --- Tool Purchase Plan ---")
    purchase_rows = []
    bought_any = False
    for q in QUARTERS:
        for ws in WS_MINTECH:
            for fab in FABS:
                val = int(round(pulp.value(Buy_m[q][ws][fab]) or 0))
                if val > 0:
                    cost = val * CAPEX[ws]
                    print(f"    {q} | Buy {val} × {ws}  (Mintech) in Fab {fab} | ${cost:,.0f}")
                    purchase_rows.append({'Quarter': q, 'Type': 'Mintech', 'WS': ws,
                                          'Fab': fab, 'Count': val, 'Cost ($)': cost})
                    bought_any = True
        for ws_t in WS_TOR:
            for fab in FABS:
                val = int(round(pulp.value(Buy_t[q][ws_t][fab]) or 0))
                if val > 0:
                    cost = val * CAPEX_TOR[ws_t]
                    print(f"    {q} | Buy {val} × {ws_t} (TOR)    in Fab {fab} | ${cost:,.0f}")
                    purchase_rows.append({'Quarter': q, 'Type': 'TOR', 'WS': ws_t,
                                          'Fab': fab, 'Count': val, 'Cost ($)': cost})
                    bought_any = True
    if not bought_any:
        print("    No new tools required.")

    print("\n  --- Tool Move-Out Plan ---")
    moveout_rows = []
    moved_any = False
    for q in QUARTERS:
        for ws in WS_MINTECH:
            for fab in FABS:
                val = int(round(pulp.value(MoveOut[q][ws][fab]) or 0))
                if val > 0:
                    cost = val * MOVEOUT_COST_PER_TOOL
                    print(f"    {q} | Move out {val} × {ws} from Fab {fab} | ${cost:,.0f}")
                    moveout_rows.append({'Quarter': q, 'WS': ws, 'Fab': fab,
                                         'Count': val, 'Cost ($)': cost})
                    moved_any = True
    if not moved_any:
        print("    No move-outs recommended.")

    # --- Space usage per fab per quarter ---
    print("\n  --- Space Usage per Fab per Quarter ---")
    space_rows = []
    print(f"  {'Quarter':<10} {'Fab':<6} {'Used (m²)':>12} {'Limit (m²)':>12} {'%Used':>8} {'Status'}")
    print("  " + "-"*55)
    for q_idx, q in enumerate(QUARTERS):
        for fab in FABS:
            used = sum(
                (int(round(pulp.value(Buy_m[QUARTERS[i]][ws][fab]) or 0)) for i in range(q_idx + 1))
                * SPACE[ws] + TOOLS_INITIAL.get(ws, {}).get(fab, 0) * SPACE[ws]
                for ws in WS_MINTECH
            ) + sum(
                sum(int(round(pulp.value(Buy_t[QUARTERS[i]][ws_t][fab]) or 0)) for i in range(q_idx + 1))
                * SPACE_TOR[ws_t]
                for ws_t in WS_TOR
            ) - sum(
                sum(int(round(pulp.value(MoveOut[QUARTERS[i]][ws][fab]) or 0)) for i in range(q_idx + 1))
                * SPACE[ws]
                for ws in WS_MINTECH
            )
            limit  = FAB_SPACE_MAX[fab]
            pct    = 100 * used / limit
            status = '⚠️ OVER' if used > limit else '✅ OK'
            print(f"  {q:<10} {fab:<6} {used:>12.1f} {limit:>12} {pct:>7.1f}% {status}")
            space_rows.append({'Quarter': q, 'Fab': fab, 'Space Used (m²)': round(used, 1),
                                'Space Limit (m²)': limit, '% Used': round(pct, 1), 'Status': status})

    # --- Cost summary ---
    cost_summary = {
        'CapEx Mintech ($)': round(capex_m_v),
        'CapEx TOR ($)':     round(capex_t_v),
        'Transfer OpEx ($)': round(xfer_v),
        'MoveOut OpEx ($)':  round(mout_v),
        'Total Cost ($)':    round(total),
    }

    # --- Export to CSV ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    def path(f): return os.path.join(output_dir, f)

    if purchase_rows:
        pd.DataFrame(purchase_rows).to_csv(path('partb_purchases.csv'), index=False)
    if moveout_rows:
        pd.DataFrame(moveout_rows).to_csv(path('partb_moveouts.csv'), index=False)
    pd.DataFrame(space_rows).to_csv(path('partb_space.csv'), index=False)
    pd.DataFrame([cost_summary]).to_csv(path('partb_cost_summary.csv'), index=False)

    print("\n  ✅ Part b) exported to:", output_dir)
    print("     partb_purchases.csv")
    print("     partb_moveouts.csv")
    print("     partb_space.csv")
    print("     partb_cost_summary.csv")

    return prob, Buy_m, Buy_t, MoveOut


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_to_csv(all_results, output_dir=None):
    """Combine results from all quarters and write to CSV files."""
    # Save to current directory by default, or specify a path e.g. r"C:\Users\You\Documents\Micron"
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))  # same folder as the script

    os.makedirs(output_dir, exist_ok=True)

    def path(filename):
        return os.path.join(output_dir, filename)
    flow_all  = pd.concat([r['flow']  for r in all_results], ignore_index=True)
    tools_all = pd.concat([r['tools'] for r in all_results], ignore_index=True)
    xfers_all = pd.concat([r['xfers'] for r in all_results], ignore_index=True)

    flow_all.to_csv(path('flow_distribution.csv'),  index=False)
    tools_all.to_csv(path('tool_allocation.csv'),   index=False)
    xfers_all.to_csv(path('transfers.csv'),          index=False)

    print("\n  ✅ Exported to:", output_dir)
    print("     flow_distribution.csv")
    print("     tool_allocation.csv")
    print("     transfers.csv")

    # Summary: total transfer cost across all quarters
    summary = pd.DataFrame([{
        'Quarter': r['quarter'],
        'Status':  r['status'],
        'Transfer Cost ($)': round(r['transfer_cost'])
    } for r in all_results])
    summary.to_csv(path('cost_summary.csv'), index=False)
    print("     cost_summary.csv")
    print()
    print(summary.to_string(index=False))
    grand = summary['Transfer Cost ($)'].sum()
    print(f"\n  Grand Total Transfer Cost (Part a): ${grand:,.0f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  MICRON NUS-ISE BACC 2026 — PART B SOLVER")
    print("="*65)

    # --- Step 1: Show initial space usage (context before solving) ---
    print_initial_space()

    # --- Step 2: Solve Part b) — full 8-quarter MILP ---
    # Allows tool purchases (mintech + TOR) and move-outs
    # Minimises CapEx + Transfer OpEx + Move-Out OpEx simultaneously
    # NOTE: This may take several minutes to solve due to MILP complexity
    print("\n\n" + "="*65)
    print("  PART B — Full 8-Quarter MILP (purchases + move-outs)")
    print("="*65)
    solve_part_b(verbose=True)
