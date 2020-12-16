##
## Training:   until 2015
## Validation: 2016-2017
## Test:       2018-2019
## 
DATE1 = '2016-01-01'
DATE2 = '2018-01-01'

## TODO: We should review those variables one by one and check the direction
## in which we expect them to forecast future returns.
## I have cheated and looked at the sign of the information ratio (in the comments)
## of a long-short quintile strategy built with each of them.

## TODO: I should have done that on the in-sample period...

signs = {
  'Advt_12M_Usd':                    -1.0,   #   -1.1
  'Advt_3M_Usd':                     -1.0,   #   -1.2
  'Advt_6M_Usd':                     -1.0,   #   -1.2
  'Asset_Turnover':                   1.0,   #  0.017
  'Bb_Yld':                          -1.0,   #  -0.46
  'Bv':                              -1.0,   #   -1.2
  'Capex_Ps_Cf':                     -1.0,   #  -0.71
  'Capex_Sales':                     -1.0,   #  -0.31
  'Cash_Div_Cf':                     -1.0,   #  -0.78
  'Cash_Per_Share':                  -1.0,   #  -0.82
  'Cf_Sales':                        -1.0,   #  -0.96
  'Debtequity':                      -1.0,   #   -0.2
  'Div_Yld':                         -1.0,   #  -0.55
  'Dps':                             -1.0,   #  -0.84
  'Ebit_Bv':                         -1.0,   #  -0.85
  'Ebit_Noa':                        -1.0,   #  -0.88
  'Ebit_Oa':                         -1.0,   #  -0.92
  'Ebit_Ta':                         -1.0,   #  -0.91
  'Ebitda_Margin':                   -1.0,   #   -0.8
  'Eps':                             -1.0,   #   -1.2
  'Eps_Basic':                       -1.0,   #   -1.2
  'Eps_Basic_Gr':                    -1.0,   #  -0.65
  'Eps_Contin_Oper':                 -1.0,   #   -1.3
  'Eps_Dil':                         -1.0,   #   -1.2
  'Ev':                              -1.0,   #   -1.4
  'Ev_Ebitda':                       -1.0,   #  -0.39
  'Fa_Ci':                            1.0,   #  0.098
  'Fcf':                             -1.0,   #  -0.99
  'Fcf_Bv':                          -1.0,   #  -0.61
  'Fcf_Ce':                          -1.0,   #  -0.55
  'Fcf_Margin':                      -1.0,   #  -0.82
  'Fcf_Noa':                         -1.0,   #  -0.72
  'Fcf_Oa':                          -1.0,   #   -0.8
  'Fcf_Ta':                          -1.0,   #  -0.83
  'Fcf_Tbv':                         -1.0,   #  -0.19
  'Fcf_Toa':                         -1.0,   #   -0.5
  'Fcf_Yld':                         -1.0,   #  -0.13
  'Free_Ps_Cf':                      -1.0,   #  -0.82
  'Int_Rev':                         -1.0,   #  -0.51
  'Interest_Expense':                -1.0,   #  -0.48
  'Mkt_Cap_12M_Usd':                 -1.0,   #   -1.4
  'Mkt_Cap_3M_Usd':                  -1.0,   #   -1.4
  'Mkt_Cap_6M_Usd':                  -1.0,   #   -1.4
  'Mom_11M_Usd':                     -1.0,   #  -0.61
  'Mom_5M_Usd':                      -1.0,   #  -0.57
  'Mom_Sharp_11M_Usd':               -1.0,   #  -0.65
  'Mom_Sharp_5M_Usd':                -1.0,   #  -0.56
  'Nd_Ebitda':                        1.0,   #   0.13
  'Net_Debt':                        -1.0,   #  -0.32
  'Net_Debt_Cf':                     -1.0,   #  -0.64
  'Net_Margin':                      -1.0,   #   -1.1
  'Netdebtyield':                    -1.0,   #  -0.47
  'Ni':                              -1.0,   #   -1.2
  'Ni_Avail_Margin':                 -1.0,   #   -1.1
  'Ni_Oa':                           -1.0,   #  -0.98
  'Ni_Toa':                          -1.0,   #  -0.72
  'Noa':                             -1.0,   #  -0.95
  'Oa':                              -1.0,   #  -0.94
  'Ocf':                             -1.0,   #   -1.1
  'Ocf_Bv':                          -1.0,   #  -0.63
  'Ocf_Ce':                          -1.0,   #  -0.41
  'Ocf_Margin':                      -1.0,   #  -0.85
  'Ocf_Noa':                         -1.0,   #  -0.58
  'Ocf_Oa':                          -1.0,   #  -0.48
  'Ocf_Ta':                          -1.0,   #  -0.46
  'Ocf_Tbv':                         -1.0,   #  -0.67
  'Ocf_Toa':                         -1.0,   #  -0.23
  'Op_Margin':                       -1.0,   #   -1.0
  'Op_Prt_Margin':                   -1.0,   #  -0.63
  'Oper_Ps_Net_Cf':                  -1.0,   #   -1.0
  'Pb':                              -1.0,   #   -1.1
  'Pe':                              -1.0,   #  -0.64
  'Ptx_Mgn':                         -1.0,   #   -1.1
  'Recurring_Earning_Total_Assets':  -1.0,   #   -1.1
  'Return_On_Capital':               -1.0,   #  -0.93
  'Rev':                             -1.0,   #  -0.88
  'Roa':                             -1.0,   #   -1.0
  'Roc':                             -1.0,   #  -0.94
  'Roce':                            -1.0,   #  -0.61
  'Roe':                             -1.0,   #  -0.99
  'Sales_Ps':                        -1.0,   #  -0.58
  'Share_Turn_12M':                   1.0,   # 0.0015
  'Share_Turn_3M':                    1.0,   #  0.042
  'Share_Turn_6M':                    1.0,   #  0.038
  'Ta':                              -1.0,   #   -1.0
  'Tev_Less_Mktcap':                 -1.0,   # -0.081
  'Tot_Debt_Rev':                    -1.0,   #  -0.16
  'Total_Capital':                   -1.0,   #   -1.1
  'Total_Debt':                      -1.0,   #  -0.62
  'Total_Debt_Capital':              -1.0,   # -0.079
  'Total_Liabilities_Total_Assets':  -1.0,   #   -0.1
  'Vol1Y_Usd':                        1.0,   #   0.53
  'Vol3Y_Usd':                        1.0,   #   0.56
}
