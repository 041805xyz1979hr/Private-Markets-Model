import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_cash_flows(commitments, call_period, hold_period, distribution_period, years,
                        target_allocation, return_rate, income_percent):
    # Special handling for (n,0,0) funds: no hold, no distribution.
    if hold_period == 0 and distribution_period == 0:
        # 1) Commit 100% in Year 1
        capital_calls = np.zeros(years)
        capital_calls[0] = target_allocation

        # 2) No principal distributions
        distributions = np.zeros(years)

        # 3) NIC with compounding on the full draw
        NIC = np.zeros(years)
        for t in range(years):
            if t == 0:
                NIC[t] = capital_calls[t]
            else:
                NIC[t] = NIC[t-1] * (1 + return_rate) + capital_calls[t]

        # 4) Income = percentage of end-year NAV (if you still want to pay it out)
        income = NIC * (income_percent / 100.0)

        # 5) Cash position: uncalled capital + any income paid
        cumulative_calls  = np.cumsum(capital_calls)
        cumulative_income = np.cumsum(income)
        cash_position     = target_allocation - cumulative_calls + cumulative_income

        return NIC, capital_calls, distributions, income, cash_position



    # Normal simulation for other structures:
    capital_calls = np.zeros(years)
    distributions = np.zeros(years)
    
    # Calculate capital calls by averaging over the call period.
    for t in range(years):
        start_year = max(0, t - call_period + 1)
        capital_calls[t] = np.sum(commitments[start_year:t+1]) / call_period
    
    # Schedule distributions for each year's commitment.
    for k in range(years):
        start_dist = k + call_period + hold_period
        if start_dist < years and distribution_period > 0:
            annual_dist = commitments[k] / distribution_period
            for t in range(start_dist, min(start_dist + distribution_period, years)):
                distributions[t] += annual_dist

    # Compute NIC (Net Invested Capital) with compounding.
    NIC = np.zeros(years)
    for t in range(years):
        if t == 0:
            NIC[t] = capital_calls[t] - distributions[t]
        else:
            NIC[t] = NIC[t-1] * (1 + return_rate) + capital_calls[t] - distributions[t]
    
    income = NIC * (income_percent / 100)
    cumulative_calls = np.cumsum(capital_calls)
    cumulative_distributions = np.cumsum(distributions)
    cumulative_income = np.cumsum(income)
    
    cash_position = target_allocation - cumulative_calls + cumulative_distributions + cumulative_income
    return NIC, capital_calls, distributions, income, cash_position


from scipy.optimize import minimize

def fund_forecast(target_allocation,
                  call_period, hold_period, distribution_period,
                  years, return_rate, income_percent):
    """
    Returns (commitments, capital_calls, distributions, NIC, income, cash_position)
    for a single fund, using:
      - your existing SLSQP when (hold=0 and dist=0)
      - a new, fully‐constrained SLSQP otherwise
    """
    n, h, d, H = call_period, hold_period, distribution_period, years
    T, r, inc = target_allocation, return_rate, income_percent

    # --------------------------------------------------------------------
    # 1) Pure (n,0,0) funds: keep your existing SLSQP block unmodified
    # --------------------------------------------------------------------
    if h == 0 and d == 0:
        alpha = 1e-3
        def objective_n000(x):
            comm = np.concatenate((x, np.zeros(H - n)))
            NIC_, _, _, _, cash_ = simulate_cash_flows(
                comm, n, h, d, H, T, r, inc
            )
            return (NIC_[-1] - T)**2 - alpha * x[0]

        def cons_n000(x):
            comm = np.concatenate((x, np.zeros(H - n)))
            _, _, _, _, cash_ = simulate_cash_flows(
                comm, n, h, d, H, T, r, inc
            )
            return cash_                # enforces cash ≥ 0 each year

        x0    = np.full(n, T/n)
        bnds  = [(0, T)] * n
        cons  = [{'type':'ineq', 'fun': cons_n000}]
        sol   = minimize(objective_n000, x0,
                         method='SLSQP',
                         bounds=bnds,
                         constraints=cons)
        x_opt = sol.x
        commitments = np.concatenate((x_opt, np.zeros(H - n)))

        NIC, cap_calls, dists, income, cash = simulate_cash_flows(
            commitments, n, h, d, H, T, r, inc
        )
        return commitments, cap_calls, dists, NIC, income, cash

    # --------------------------------------------------------------------
    # 2) (h>0 or d>0) funds: single SLSQP with vector‐ineq constraints
    # --------------------------------------------------------------------
    # we’ll optimise over x = [c1, c2]
    def objective(x):
        # any increasing surrogate; here sum(c1,c2)
        return x[0] + x[1]

    def constraints(x):
        # build full schedule [c1,c2,c2,...]
        c1, c2 = x
        commits = np.concatenate(([c1], np.full(H - 1, c2)))
        NIC_, _, _, _, cash_ = simulate_cash_flows(
            commits, n, h, d, H, T, r, inc
        )
        # return a vector: first all years’ cash, then final NIC−T
        return np.concatenate([cash_, [NIC_[-1] - T]])

    # initial guess & bounds
    x0   = np.full(2, T / n)
    bnds = [(0, T), (0, T)]

    sol = minimize(objective, x0,
                   method='SLSQP',
                   bounds=bnds,
                   constraints=[{'type':'ineq', 'fun': constraints}],
                   options={'ftol':1e-9, 'maxiter':100})

    c1_opt, c2_opt = sol.x
    commitments = np.concatenate(([c1_opt], np.full(H - 1, c2_opt)))

    NIC, cap_calls, dists, income, cash = simulate_cash_flows(
        commitments, n, h, d, H, T, r, inc
    )
    return commitments, cap_calls, dists, NIC, income, cash


def add_total_row(df, year_columns):
    sum_allocation = df['Allocation'].sum()
    sum_allocation_percent = df['Allocation (%)'].sum()
    sums_by_year = df[year_columns].sum()
    total_row = [
        'Total', '', '', '',
        sum_allocation, sum_allocation_percent, '', '', ''
    ] + sums_by_year.tolist()
    total_df = pd.DataFrame([total_row], columns=df.columns)
    return pd.concat([df, total_df], ignore_index=True)


def multi_fund_forecast(funds_dict, years):
    metadata_columns = [
        'Fund Name','Liquidity','Status','Asset Class',
        'Allocation','Allocation (%)','Call Period',
        'Hold Period','Distribution Period'
    ]
    year_columns = [f'Year {i}' for i in range(1, years + 1)]
    
    # Step 1: Extract each fund’s parameters
    fund_params = []
    total_allocation = sum(f['allocation'] for f in funds_dict.values())
    for name, f in funds_dict.items():
        fund_params.append({
            'name': name,
            'target_allocation': f['allocation'],
            'call_period':     f['call_period'],
            'hold_period':     f['hold_period'],
            'distribution_period': f['distribution_period'],
            'return_rate':     f.get('return_rate', 0.0),
            'income_percent':  f.get('income_percent', 0.0),
            'liquidity':       f['liquidity'],
            'status':          f.get('status','Active'),
            'asset_class':     f['asset_class'],
            'allocation_percent': f['allocation_percent']
        })
    
    # We will store each fund’s *final* commitment schedule (Year1..YearN) here
    all_commitments = []
    all_cap_calls   = []
    all_distributions = []
    all_NIC         = []
    all_income      = []
    all_cash_pos    = []
    
    # Step 2: For each fund, decide whether to use closed‐form `fund_forecast`
    #         or fall back to (n,0,0) SLSQP or other logic.
    for fund in fund_params:
        n = fund['call_period']
        h = fund['hold_period']
        d = fund['distribution_period']
        T = fund['target_allocation']
        r = fund['return_rate']
        inc_pct = fund['income_percent']
        
        # If this is a pure (n,0,0) structure, we can keep your old SLSQP block:
        if (h == 0 and d == 0):
            # — EXACTLY THE CODE YOU ALREADY HAD for (n,0,0) funds —
            alpha = 1e-3
            def obj_n000(x):
                comm = np.concatenate((x, np.zeros(years - n)))
                NIC_, _, _, _, cash_ = simulate_cash_flows(
                    comm, n, h, d, years,
                    T, r, inc_pct
                )
                return (NIC_[-1] - T)**2 - alpha*x[0]
            
            def cons_n000(x):
                comm = np.concatenate((x, np.zeros(years - n)))
                _, _, _, _, cash_ = simulate_cash_flows(
                    comm, n, h, d, years,
                    T, r, inc_pct
                )
                # enforce cash ≥ 0 for all t
                return cash_
            
            x0 = np.full(n, T/n)
            bnds = [(0, T)] * n
            cons = [{'type':'ineq','fun': cons_n000}]
            sol = minimize(obj_n000, x0, method='SLSQP', bounds=bnds, constraints=cons)
            x_opt = sol.x
            commitments = np.concatenate((x_opt, np.zeros(years - n)))
            
            NIC_, cap_calls, distributions, income, cash_pos = simulate_cash_flows(
                commitments, n, h, d, years, T, r, inc_pct
            )
        
        # Otherwise (h>0 or d>0), *call* your new fund_forecast directly:
        else:
            commitments, cap_calls, distributions, NIC_, income, cash_pos = \
                fund_forecast(T, n, h, d, years, r, inc_pct)
        
        # ------------- collect everything -------------
        all_commitments.append(commitments)
        all_cap_calls.append(cap_calls)
        all_distributions.append(distributions)
        all_NIC.append(NIC_)
        all_income.append(income)
        all_cash_pos.append(cash_pos)
        
    adjustable = [i for i, f in enumerate(fund_params)
                  if f['hold_period'] > 0 or f['distribution_period'] > 0]
    allocs     = [fund_params[i]['allocation_percent'] for i in adjustable]
    sum_allocs = sum(allocs)

    weights   = { i: allocs[j]/sum_allocs 
               for j,i in enumerate(adjustable) }
    
    orig_commitments = [comm.copy() for comm in all_commitments]
    agg_cash_before   = np.sum([all_cash_pos[i] for i in adjustable], axis=0)
    
    # Extract Year-1 for every fund:
    year1_commits = [comm[0] for comm in all_commitments]
    sum_year1 = sum(year1_commits)
    sum_target = sum(f['target_allocation'] for f in fund_params) - (0.5 * sum(f['target_allocation'] for f in fund_params))

    # Simple dump:
    
    max_iter = 10
    curr = 1
    
    while sum_year1 < sum_target and max_iter > curr:
        if adjustable:
            # Aggregate cash-position over time for just those funds
            agg_cash = np.sum([all_cash_pos[i] for i in adjustable], axis=0)
            # Find the absolute minimum cash position
            min_cash = np.min(agg_cash)
            
            total_needed = sum_target - sum_year1
            total_adj    = max(0.0, total_needed)
                
            for i in adjustable:
                all_commitments[i][0] += total_adj * weights[i]
            # Use the full trough to boost year-1 commitments (so new trough = 0)
            adjustment_total = min_cash
            per_fund_adj = adjustment_total

            # Bump each fund's first-year commitment and re-simulate
            for i in adjustable:
                all_commitments[i][0] += total_adj * weights[i]

            # Bump each fund's first-year commitment and re-simulate
            for i in adjustable:
                f = fund_params[i]
                NIC_, cap_calls, distributions, income, cash_pos = simulate_cash_flows(
                    all_commitments[i],
                    f['call_period'], f['hold_period'], f['distribution_period'],
                    years, f['target_allocation'], f['return_rate'], f['income_percent']
                )
                all_cap_calls[i]     = cap_calls
                all_distributions[i] = distributions
                all_NIC[i]           = NIC_
                all_income[i]        = income
                all_cash_pos[i]      = cash_pos

        new_commitments = all_commitments
        agg_cash_after  = np.sum([all_cash_pos[i] for i in adjustable], axis=0)

        year1_commits = [comm[0] for comm in new_commitments]
        sum_year1     = sum(year1_commits)
        st.write(sum_year1)
        curr += 1
        
    
    #st.subheader("Year-1 Commitments Before vs After")
    #st.table(pd.DataFrame({
    #        "Fund":       [i+1 for i in adjustable],
    #       "Before c1": [orig_commitments[i][0] for i in adjustable],
    #       "After  c1": [new_commitments[i][0] for i in adjustable],
    #   }))
        
    #df_agg_cash = pd.DataFrame({
    #       'Year':    [f'Year {i}' for i in range(1, years+1)],
    #       'Before':  agg_cash_before,
    #       'After':   agg_cash_after
    #   })
    #st.table(df_agg_cash)
    


    
    # Step 3: Aggregate across *all* funds into DataFrames
    # (metadata + Year1..YearN for each fund)
    commitments_data = []
    calls_data       = []
    dist_data        = []
    nic_data         = []
    income_data      = []
    returns_data     = []
    cashpos_data     = []
    
    for i, fund in enumerate(fund_params):
        meta = [
            fund['name'],
            fund['liquidity'],
            fund['status'],
            fund['asset_class'],
            fund['target_allocation'],
            fund['allocation_percent'],
            fund['call_period'],
            fund['hold_period'],
            fund['distribution_period']
        ]
        # append Year1..YearN arrays
        commitments_data.append(meta + list(all_commitments[i]))
        calls_data.append      (meta + list(all_cap_calls[i]))
        dist_data.append       (meta + list(all_distributions[i]))
        nic_data.append        (meta + list(all_NIC[i]))
        income_data.append     (meta + list(all_income[i]))
        
        # “Returns” = NIC[t-1]*r (year 0 = 0)
        returns_vec = np.zeros(years)
        for t in range(1, years):
            returns_vec[t] = all_NIC[i][t-1] * fund['return_rate']
        returns_data.append(meta + list(returns_vec))
        
        cashpos_data.append    (meta + list(all_cash_pos[i]))
    
    # Build pandas DataFrames
    df_comm    = pd.DataFrame(commitments_data, columns=metadata_columns + year_columns)
    df_calls   = pd.DataFrame(calls_data,       columns=metadata_columns + year_columns)
    df_dist    = pd.DataFrame(dist_data,        columns=metadata_columns + year_columns)
    df_nic     = pd.DataFrame(nic_data,         columns=metadata_columns + year_columns)
    df_income  = pd.DataFrame(income_data,      columns=metadata_columns + year_columns)
    df_returns = pd.DataFrame(returns_data,     columns=metadata_columns + year_columns)
    df_cashpos = pd.DataFrame(cashpos_data,     columns=metadata_columns + year_columns)
    
    # Step 4: Append “Total” row to each
    df_comm    = add_total_row(df_comm,    year_columns)
    df_calls   = add_total_row(df_calls,   year_columns)
    df_dist    = add_total_row(df_dist,    year_columns)
    df_nic     = add_total_row(df_nic,     year_columns)
    df_income  = add_total_row(df_income,  year_columns)
    df_returns = add_total_row(df_returns, year_columns)
    df_cashpos = add_total_row(df_cashpos, year_columns)
    
    
    return df_comm, df_calls, df_dist, df_nic, df_income, df_returns, df_cashpos


def style_dataframe(df):
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    cmap = sns.light_palette("green", as_cmap=True)
    styled_df = df.style.format("{:,.0f}", subset=numeric_cols) \
                        .background_gradient(cmap=cmap, subset=numeric_cols, axis=None) \
                        .set_properties(**{
                            'text-align': 'center',
                            'border': '1px solid #006400',
                            'padding': '8px',
                            'font-size': '14px'
                        }) \
                        .set_table_styles([
                            {'selector': 'th',
                             'props': [
                                 ('background-color', '#f4f4f4'),
                                 ('font-weight', 'bold'),
                                 ('text-align', 'center'),
                                 ('border', '1px solid #006400'),
                                 ('padding', '8px')
                             ]
                            }
                        ])
    return styled_df

# After style_dataframe, add:

def append_cumulative_row(df):
    # Build year_columns list exactly as you already have it in scope
    year_columns = [c for c in df.columns if c.startswith("Year ")]
    cum = df.iloc[-1][year_columns].astype(float).cumsum()
    meta_blank = ['Cumulative'] + ['']*(df.shape[1]-len(year_columns)-1)
    cum_row = meta_blank + cum.tolist()
    return pd.concat([df, pd.DataFrame([cum_row], columns=df.columns)], ignore_index=True)


from pandas import IndexSlice

def style_with_borders(df):
    styled = style_dataframe(df)  # your existing green‐gradient + formatting
    n = len(df)
    # Target the last two rows (Total and Cumulative):
    styled = styled.set_properties(
        subset=IndexSlice[[n-2, n-1], :],
        **{'border': '2px solid black'}
    )
    return styled


def main():
    st.set_page_config(layout="wide")
    st.title("Multi-Fund Forecasting App")
    st.write(
        """
        This application reads a CSV containing:
        **[Liquidity-Level, Sub-Asset Class, Capital Allocation ($), Funding Period (Y), 
        Lock-Up Period (Y), Distribuition Period (Y), Income (%), Return (%)]**,
        then runs a multi-fund capital forecast over a user-defined number of years.
        """
    )
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    forecast_years = st.number_input("Total number of forecast years:", min_value=1, value=10)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df)

        if st.button("Run Forecast"):
            funds_dict = {}
            total_allocation = df["Capital Allocation ($)"].sum()
            
            # Build a dictionary of funds from the CSV.
            for i, row in df.iterrows():
                fund_name = f"Fund {i+1}"
                allocation = float(row["Capital Allocation ($)"])
                allocation_percent = (allocation / total_allocation) * 100 if total_allocation != 0 else 0
                
                # Parse "Return (%)" safely.
                if "Growth (%)" in df.columns and pd.notna(row["Growth (%)"]):
                    return_rate = float(row["Growth (%)"]) / 100.0
                else:
                    return_rate = 0.0
                
                if "Income (%)" in df.columns and pd.notna(row["Income (%)"]):
                    income_percent = float(row["Income (%)"])
                else:
                    income_percent = 0.0
                
                
                funds_dict[fund_name] = {
                    "liquidity":           row["Liquidity-Level"],
                    "status":              "Active",
                    "asset_class":         row["Sub-Asset Class"],
                    "allocation":          allocation,
                    "allocation_percent":  allocation_percent,
                    "call_period":         int(row["Funding Period (Y)"]),
                    "hold_period":         int(row["Lock-Up Period (Y)"]),
                    "distribution_period": int(row["Distribuition Period (Y)"]),
                    "income_percent":      income_percent,
                    "return_rate":         return_rate
                }
            
            # Run the multi-fund forecast.
            df_comm, df_calls, df_dist, df_nic, df_income, df_returns, df_cash_position = multi_fund_forecast(funds_dict, forecast_years)
            
            # Round numeric columns.
            for df_ in [df_comm, df_calls, df_dist, df_nic, df_income, df_returns]:
                numeric_cols = df_.select_dtypes(include=['float', 'int']).columns
                df_[numeric_cols] = df_[numeric_cols].round(0).astype(int)

            
            
            
            
            
            # Display Tables.
            st.subheader("Capital Commitments Table")
            st.dataframe(style_dataframe(df_comm), use_container_width=True)
            
        
            
            #  Capital Calls
            st.subheader("Capital Calls Table")
            df_calls = append_cumulative_row(df_calls)
            st.dataframe(style_with_borders(df_calls), use_container_width=True)

            #  Income
            st.subheader("Income Table")
            df_income = append_cumulative_row(df_income)
            st.dataframe(style_with_borders(df_income), use_container_width=True)

            #  Growth
            st.subheader("Growth Table")
            df_returns = append_cumulative_row(df_returns)
            st.dataframe(style_with_borders(df_returns), use_container_width=True)

            
            st.subheader("Net Invested Capital Table")
            st.dataframe(style_dataframe(df_nic), use_container_width=True)
            
            #  Distributions
            st.subheader("Distributions Table")
            df_dist = append_cumulative_row(df_dist)
            st.dataframe(style_with_borders(df_dist), use_container_width=True)
            
            st.subheader("Cash Position Table")
            st.dataframe(style_dataframe(df_cash_position.iloc[[-1]]), use_container_width=True)
            
            import altair as alt

            # --- 1) Build year_columns once more ---
            year_columns = [f'Year {i}' for i in range(1, forecast_years+1)]

            # --- 2) Aggregate Portfolio Metrics table ---
            metrics1 = [
                "Target Allocation",
                "Total Capital Calls",
                "Total Income",
                "Total Growth",
                "Total Distributions",
                "Net Invested Capital",
                "Cash Position"
            ]
            data1 = {
                "Metric": metrics1,
                **{yr: [
                    total_allocation,                               # Commits
                    df_calls   .iloc[-1][yr],                       # Capital Calls
                    df_income  .iloc[-1][yr],                       # Income
                    df_returns .iloc[-1][yr],                       # Growth
                    df_dist    .iloc[-1][yr],                       # Distributions
                    df_nic     .iloc[-1][yr],                       # NIC
                    df_cash_position .iloc[-1][yr]                        # Cash Position
                ] for yr in year_columns}
            }
            df_portfolio = pd.DataFrame(data1)

            st.subheader("Aggregate Portfolio Metrics")
            st.dataframe(style_dataframe(df_portfolio), use_container_width=True)

            


            # --- 3) Portfolio Balance Sheet table ---
            metrics2 = [
                "Private Markets Portfolio",
                "Cash Position",
                "Total"
            ]
            data2 = {
                "Metric": metrics2,
                **{yr: [
                    df_nic   .iloc[-1][yr],                      # NAV
                    df_cash_position.iloc[-1][yr],                     # Cash
                    df_nic   .iloc[-1][yr] + df_cash_position.iloc[-1][yr]  # Total
                ] for yr in year_columns}
            }
            df_balance = pd.DataFrame(data2)

            st.subheader("Portfolio Balance Sheet")
            st.dataframe(style_dataframe(df_balance), use_container_width=True)
            
            year_columns = [f'Year {i}' for i in range(1, forecast_years+1)]

            metrics3 = ["Principal", "Income", "Growth", "Total"]
            data3 = {
                "Metric": metrics3,
                **{
                    yr: [
                        total_allocation,                        # Principal = constant 20M
                        df_income .iloc[-1][yr],                # Total Income in yr
                        df_returns.iloc[-1][yr],                # Total Growth in yr
                        total_allocation    
                        + df_income .iloc[-1][yr]                  # then sum: principal+income+growth
                    + df_returns.iloc[-1][yr]
                    + 0  # you can add distributions here if you want them included
                    ]
                    for yr in year_columns
                }
            }

            # Build and display
            df_breakdown = pd.DataFrame(data3)
            st.subheader("Principal / Income / Growth / Total Roll-Forward")
            st.dataframe(style_dataframe(df_breakdown), use_container_width=True)

            # --- Chart B: Stacked area of NAV, Cash, Total ---
            df_stack = df_balance.melt(id_vars="Metric", var_name="Year", value_name="Amount")
            
            chart_lines = (
                            alt.Chart(df_stack)
                            .mark_line(point=True)
                            .encode(
                                x="Year:O",
                                y="Amount:Q",
                                color="Metric:N",
                                tooltip=["Metric","Amount"]
                            )
                            .properties(title="NAV & Cash Composition Over Time (Line)")
                        )

            st.altair_chart(chart_lines, use_container_width=True)
            
            # --- 4.8) Principal / Income / Growth / Total breakdown ---
            # Reuse your total_years list
            



            import altair as alt
            import numpy as np

            # ------------------------------
            # Generate Aggregated Graphs
            # ------------------------------
            total_years = [f'Year {i}' for i in range(1, forecast_years + 1)]
            totals = {
                'Commitments': df_comm.iloc[-1][total_years].astype(float).values,
                'Capital Calls': df_calls.iloc[-1][total_years].astype(float).values,
                'Distributions': df_dist.iloc[-1][total_years].astype(float).values,
                'NIC': df_nic.iloc[-1][total_years].astype(float).values,
                'Income': df_income.iloc[-1][total_years].astype(float).values,
                'Returns': df_returns.iloc[-1][total_years].astype(float).values
            }
            df_totals = pd.DataFrame({
                'Year': np.arange(1, forecast_years + 1),
                **totals
            })

            # Calculate additional series
            df_totals['Net Cash Flow'] = df_totals['Distributions'] - df_totals['Capital Calls']
            df_totals['Cumulative Net Cash Flow'] = df_totals['Net Cash Flow'].cumsum()
            cum_calls = df_totals['Capital Calls'].cumsum()
            cum_dists = df_totals['Distributions'].cumsum()
            df_totals['MOIC'] = (cum_dists + df_totals['NIC']) / cum_calls.replace(0, np.nan)

            col1, col2 = st.columns(2)

            with col1:
                # Plot 1: Commitments vs Calls vs Distributions
                chart1 = alt.Chart(df_totals).transform_fold(
                    ['Commitments', 'Capital Calls', 'Distributions'],
                    as_=['Category', 'Amount']
                ).mark_line(point=True).encode(
                    x='Year:O',
                    y='Amount:Q',
                    color='Category:N'
                ).properties(
                    title="Commitments, Capital Calls, Distributions"
                )
                st.altair_chart(chart1, use_container_width=True)

                # Plot 3: Income vs Returns
                chart3 = alt.Chart(df_totals).transform_fold(
                    ['Income', 'Returns'],
                    as_=['Category', 'Amount']
                ).mark_line(point=True).encode(
                    x='Year:O',
                    y='Amount:Q',
                    color='Category:N'
                ).properties(
                    title="Income vs Returns"
                )
                st.altair_chart(chart3, use_container_width=True)

                # Plot 5: NIC by Fund in Final Year
                df_nic_no_total = df_nic.iloc[:-1]
                final_col = f'Year {forecast_years}'
                chart5 = alt.Chart(df_nic_no_total).mark_bar().encode(
                    x=alt.X('Fund Name:N', sort=None),
                    y=alt.Y(f'{final_col}:Q', title='NIC'),
                    tooltip=['Fund Name', final_col]
                ).properties(
                    title=f"NIC by Fund in Year {forecast_years}"
                )
                st.altair_chart(chart5, use_container_width=True)
                
                # -----------------------------------------
                # Show NIC with and without compounding
                # -----------------------------------------
                # Compute “linear NIC” = cumulative calls – cumulative distributions
                df_totals['Cumulative Calls'] = df_totals['Capital Calls'].cumsum()
                df_totals['Cumulative Distributions'] = df_totals['Distributions'].cumsum()
                df_totals['NIC_linear'] = df_totals['Cumulative Calls'] - df_totals['Cumulative Distributions']

                # Build an Altair line chart comparing actual vs. linear NIC
                nic_compare = (
                    alt.Chart(df_totals)
                    .transform_fold(
                        ['NIC', 'NIC_linear'],
                        as_=['Scenario', 'Value']
                    )
                    .mark_line(point=True)
                    .encode(
                        x='Year:O',
                        y='Value:Q',
                        color='Scenario:N'
                    )
                    .properties(
                        title='Actual NIC (with return) vs. Linear NIC (no return)'
                    )
                )
                st.altair_chart(nic_compare, use_container_width=True)

            with col2:
                # Plot 2: NIC Over Time with Total Commitments
                base_nic = alt.Chart(df_totals).mark_line(point=True, color='blue').encode(
                    x='Year:O',
                    y='NIC:Q'
                )
                line_comm = alt.Chart(df_totals).mark_rule(color='red', strokeDash=[4,4]).encode(
                    y='Commitments:Q',
                    tooltip=[alt.Tooltip('Commitments:Q', title='Total Commitments')]
                )
                chart2 = alt.layer(base_nic, line_comm).properties(
                    title="Net Invested Capital Over Time"
                )
                st.altair_chart(chart2, use_container_width=True)

                # Plot 4: Net Cash Flow & Cumulative Net Cash Flow
                bar_net = alt.Chart(df_totals).mark_bar(color='teal').encode(
                    x='Year:O',
                    y='Net Cash Flow:Q'
                )
                line_cum = alt.Chart(df_totals).mark_line(point=True, color='purple').encode(
                    x='Year:O',
                    y='Cumulative Net Cash Flow:Q'
                )
                chart4 = alt.layer(bar_net, line_cum).properties(
                    title="Net Cash Flow & Cumulative Net Cash Flow"
                )
                st.altair_chart(chart4, use_container_width=True)

                # Plot 6: MOIC Progression
                chart6 = alt.Chart(df_totals).mark_line(point=True, color='brown').encode(
                    x='Year:O',
                    y='MOIC:Q'
                ).properties(
                    title="MOIC Progression Over Time"
                )
                st.altair_chart(chart6, use_container_width=True)


if __name__ == "__main__":
    main()
