"""
BharatNXT Synthetic Dataset Generator
2000 rows, 41 columns, 6 personas, realistic Indian SME distributions
Run: python generate_data.py  → produces bharatnxt_survey_data.csv
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
random.seed(42)

N = 2000

# ── Persona budgets ──────────────────────────────────────────────────────────
PERSONAS = {
    "P1_TechFounder":    700,
    "P2_REBroker":       500,
    "P3_TradSME":        300,
    "P4_Consultant":     250,
    "P5_Skeptic":        150,
    "P6_PowerUser":      100,
}

def weighted_choice(options, weights, size=1):
    return np.random.choice(options, size=size, p=np.array(weights)/sum(weights))

def build_persona(persona, n):
    rows = []
    for _ in range(n):
        r = {}
        r["persona_label"] = persona

        # ── Section A: Demographics ──────────────────────────────────────
        if persona == "P1_TechFounder":
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [8,55,28,7,2])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [68,25,6,1])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [5,75,15,5])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [10,35,38,17])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [5,25,45,25])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [60,20,10,8,2])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [40,38,17,5])[0]

        elif persona == "P2_REBroker":
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [3,20,42,28,7])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [30,40,24,6])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [88,2,5,5])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [5,18,35,42])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [10,30,40,20])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [45,30,15,7,3])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [15,35,35,15])[0]

        elif persona == "P3_TradSME":
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [2,10,30,38,20])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [12,28,38,22])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [20,5,10,65])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [3,10,28,59])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [35,40,20,5])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [35,40,15,5,5])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [5,20,35,40])[0]

        elif persona == "P4_Consultant":
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [10,52,30,6,2])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [72,22,5,1])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [3,25,65,7])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [12,38,35,15])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [15,35,38,12])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [65,15,12,6,2])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [38,42,15,5])[0]

        elif persona == "P5_Skeptic":
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [5,25,35,28,7])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [35,35,22,8])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [30,30,20,20])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [5,20,38,37])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [15,35,35,15])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [40,35,15,7,3])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [5,20,35,40])[0]

        else:  # P6_PowerUser
            r["age_group"] = weighted_choice(["18-25","26-35","36-45","46-55","56+"], [3,35,40,18,4])[0]
            r["city_tier"] = weighted_choice(["Metro","Tier1","Tier2","Tier3"], [60,30,8,2])[0]
            r["business_type"] = weighted_choice(["Real Estate","IT/Software","Consulting","Other SME"], [45,40,10,5])[0]
            r["business_age"] = weighted_choice(["<1yr","1-3yr","3-7yr","7+yr"], [2,15,38,45])[0]
            r["annual_turnover"] = weighted_choice(["<25L","25L-1Cr","1Cr-5Cr",">5Cr"], [2,8,35,55])[0]
            r["decision_authority"] = weighted_choice(["Owner alone","With CA","Partner","CFO","Board"], [50,20,15,12,3])[0]
            r["adoption_cycle"] = weighted_choice(["<1week","2-4weeks","1-3months",">3months"], [45,35,15,5])[0]

        # ── Section B: Payment Behaviour ─────────────────────────────────
        if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]:
            r["payment_methods"] = weighted_choice(["NEFT/RTGS","UPI","Cheque","Credit Card","Cash","Mixed"],
                                                    [20,35,5,15,5,20])[0]
            r["monthly_payment_vol"] = weighted_choice(["<1L","1L-5L","5L-25L",">25L"], [5,20,45,30])[0]
            r["cash_flow_crisis_freq"] = weighted_choice(["Never","Rarely","Sometimes","Frequently"], [15,35,35,15])[0]
            r["has_credit_card"] = weighted_choice(["Yes_corporate","Yes_personal","No_planning","No_no_need"],
                                                    [40,40,15,5])[0]
            r["working_capital_crisis"] = weighted_choice(["Yes_severe","Yes_manageable","Ongoing","No_stable"],
                                                           [10,30,15,45])[0]
        elif persona == "P2_REBroker":
            r["payment_methods"] = weighted_choice(["NEFT/RTGS","UPI","Cheque","Credit Card","Cash","Mixed"],
                                                    [30,25,20,10,5,10])[0]
            r["monthly_payment_vol"] = weighted_choice(["<1L","1L-5L","5L-25L",">25L"], [8,30,42,20])[0]
            r["cash_flow_crisis_freq"] = weighted_choice(["Never","Rarely","Sometimes","Frequently"], [10,30,40,20])[0]
            r["has_credit_card"] = weighted_choice(["Yes_corporate","Yes_personal","No_planning","No_no_need"],
                                                    [20,45,25,10])[0]
            r["working_capital_crisis"] = weighted_choice(["Yes_severe","Yes_manageable","Ongoing","No_stable"],
                                                           [15,35,20,30])[0]
        elif persona == "P3_TradSME":
            r["payment_methods"] = weighted_choice(["NEFT/RTGS","UPI","Cheque","Credit Card","Cash","Mixed"],
                                                    [25,30,20,5,15,5])[0]
            r["monthly_payment_vol"] = weighted_choice(["<1L","1L-5L","5L-25L",">25L"], [35,40,20,5])[0]
            r["cash_flow_crisis_freq"] = weighted_choice(["Never","Rarely","Sometimes","Frequently"], [20,35,30,15])[0]
            r["has_credit_card"] = weighted_choice(["Yes_corporate","Yes_personal","No_planning","No_no_need"],
                                                    [5,30,30,35])[0]
            r["working_capital_crisis"] = weighted_choice(["Yes_severe","Yes_manageable","Ongoing","No_stable"],
                                                           [12,28,18,42])[0]
        else:  # P5_Skeptic
            r["payment_methods"] = weighted_choice(["NEFT/RTGS","UPI","Cheque","Credit Card","Cash","Mixed"],
                                                    [28,30,15,10,10,7])[0]
            r["monthly_payment_vol"] = weighted_choice(["<1L","1L-5L","5L-25L",">25L"], [15,35,38,12])[0]
            r["cash_flow_crisis_freq"] = weighted_choice(["Never","Rarely","Sometimes","Frequently"], [15,30,35,20])[0]
            r["has_credit_card"] = weighted_choice(["Yes_corporate","Yes_personal","No_planning","No_no_need"],
                                                    [20,40,25,15])[0]
            r["working_capital_crisis"] = weighted_choice(["Yes_severe","Yes_manageable","Ongoing","No_stable"],
                                                           [20,30,20,30])[0]

        # Regular payment types (multi-select encoded as counts)
        pay_types = ["vendor_pay","gst_tax","rent","utility","salary","saas_sub","insurance","loan_emi"]
        if persona in ["P1_TechFounder","P4_Consultant"]:
            probs = [0.85,0.80,0.70,0.75,0.65,0.80,0.40,0.30]
        elif persona == "P2_REBroker":
            probs = [0.90,0.85,0.80,0.70,0.55,0.35,0.50,0.40]
        elif persona == "P6_PowerUser":
            probs = [0.95,0.90,0.85,0.80,0.75,0.70,0.60,0.50]
        elif persona == "P3_TradSME":
            probs = [0.75,0.70,0.60,0.65,0.50,0.20,0.35,0.30]
        else:
            probs = [0.80,0.75,0.65,0.70,0.55,0.50,0.40,0.35]
        for pt, p in zip(pay_types, probs):
            r[f"pays_{pt}"] = int(np.random.random() < p)
        r["num_payment_types"] = sum(r[f"pays_{pt}"] for pt in pay_types)

        # Competitor tools tried
        tools = ["Razorpay_PayU","BharatPe","Instamojo","Bank_netbanking","NBFC_apps","None"]
        if persona in ["P1_TechFounder","P4_Consultant"]:
            t_probs = [0.55,0.25,0.20,0.70,0.15,0.10]
        elif persona == "P2_REBroker":
            t_probs = [0.25,0.35,0.15,0.80,0.20,0.15]
        elif persona == "P5_Skeptic":
            t_probs = [0.40,0.30,0.25,0.75,0.35,0.05]
        else:
            t_probs = [0.30,0.25,0.15,0.75,0.20,0.20]
        for t, p in zip(tools, t_probs):
            r[f"tried_{t.replace('/','_')}"] = int(np.random.random() < p)

        if persona == "P5_Skeptic":
            r["switching_trigger"] = weighted_choice(
                ["Lower_fee","Better_integration","Bank_trust","Better_support","All_in_one","NA"],
                [15,10,40,20,10,5])[0]
        elif persona in ["P1_TechFounder","P4_Consultant"]:
            r["switching_trigger"] = weighted_choice(
                ["Lower_fee","Better_integration","Bank_trust","Better_support","All_in_one","NA"],
                [20,35,10,10,20,5])[0]
        else:
            r["switching_trigger"] = weighted_choice(
                ["Lower_fee","Better_integration","Bank_trust","Better_support","All_in_one","NA"],
                [25,15,25,15,15,5])[0]

        # ── Section C: Trust & Risk ───────────────────────────────────────
        if persona == "P5_Skeptic":
            r["biggest_concern"] = weighted_choice(
                ["Data_security","Not_RBI_regulated","Hidden_charges","Platform_shutdown","CA_wont_approve","No_concerns"],
                [30,25,20,15,8,2])[0]
            r["fraud_experience"] = weighted_choice(["Yes_significant","Yes_minor","No","Prefer_not_say"],
                                                     [50,30,15,5])[0]
            r["bad_fintech_exp"] = weighted_choice(["Yes_very_cautious","Yes_open","No_bad_exp","No_exp"],
                                                    [60,20,15,5])[0]
            r["rbi_importance"] = weighted_choice([1,2,3,4,5], [2,5,10,25,58])[0]
        elif persona in ["P1_TechFounder","P4_Consultant"]:
            r["biggest_concern"] = weighted_choice(
                ["Data_security","Not_RBI_regulated","Hidden_charges","Platform_shutdown","CA_wont_approve","No_concerns"],
                [25,15,20,10,5,25])[0]
            r["fraud_experience"] = weighted_choice(["Yes_significant","Yes_minor","No","Prefer_not_say"],
                                                     [5,20,70,5])[0]
            r["bad_fintech_exp"] = weighted_choice(["Yes_very_cautious","Yes_open","No_bad_exp","No_exp"],
                                                    [5,20,60,15])[0]
            r["rbi_importance"] = weighted_choice([1,2,3,4,5], [5,15,30,30,20])[0]
        elif persona == "P3_TradSME":
            r["biggest_concern"] = weighted_choice(
                ["Data_security","Not_RBI_regulated","Hidden_charges","Platform_shutdown","CA_wont_approve","No_concerns"],
                [20,25,25,10,15,5])[0]
            r["fraud_experience"] = weighted_choice(["Yes_significant","Yes_minor","No","Prefer_not_say"],
                                                     [10,25,58,7])[0]
            r["bad_fintech_exp"] = weighted_choice(["Yes_very_cautious","Yes_open","No_bad_exp","No_exp"],
                                                    [15,20,45,20])[0]
            r["rbi_importance"] = weighted_choice([1,2,3,4,5], [3,8,20,35,34])[0]
        else:
            r["biggest_concern"] = weighted_choice(
                ["Data_security","Not_RBI_regulated","Hidden_charges","Platform_shutdown","CA_wont_approve","No_concerns"],
                [22,18,22,12,8,18])[0]
            r["fraud_experience"] = weighted_choice(["Yes_significant","Yes_minor","No","Prefer_not_say"],
                                                     [8,22,62,8])[0]
            r["bad_fintech_exp"] = weighted_choice(["Yes_very_cautious","Yes_open","No_bad_exp","No_exp"],
                                                    [10,22,52,16])[0]
            r["rbi_importance"] = weighted_choice([1,2,3,4,5], [4,10,25,32,29])[0]

        # ── Section D: Tech Readiness ─────────────────────────────────────
        if persona in ["P1_TechFounder","P4_Consultant"]:
            r["digital_comfort"] = weighted_choice(
                ["Very_comfortable","Comfortable","Cautious","Not_comfortable"], [55,35,8,2])[0]
            r["onboarding_pref"] = weighted_choice(
                ["Self_tutorial","WhatsApp_chat","Phone_agent","Inperson_visit","CA_setup"], [50,25,15,5,5])[0]
        elif persona == "P6_PowerUser":
            r["digital_comfort"] = weighted_choice(
                ["Very_comfortable","Comfortable","Cautious","Not_comfortable"], [65,28,5,2])[0]
            r["onboarding_pref"] = weighted_choice(
                ["Self_tutorial","WhatsApp_chat","Phone_agent","Inperson_visit","CA_setup"], [55,25,12,3,5])[0]
        elif persona == "P3_TradSME":
            r["digital_comfort"] = weighted_choice(
                ["Very_comfortable","Comfortable","Cautious","Not_comfortable"], [8,25,42,25])[0]
            r["onboarding_pref"] = weighted_choice(
                ["Self_tutorial","WhatsApp_chat","Phone_agent","Inperson_visit","CA_setup"], [10,20,25,15,30])[0]
        elif persona == "P5_Skeptic":
            r["digital_comfort"] = weighted_choice(
                ["Very_comfortable","Comfortable","Cautious","Not_comfortable"], [20,35,30,15])[0]
            r["onboarding_pref"] = weighted_choice(
                ["Self_tutorial","WhatsApp_chat","Phone_agent","Inperson_visit","CA_setup"], [20,25,25,10,20])[0]
        else:
            r["digital_comfort"] = weighted_choice(
                ["Very_comfortable","Comfortable","Cautious","Not_comfortable"], [30,45,20,5])[0]
            r["onboarding_pref"] = weighted_choice(
                ["Self_tutorial","WhatsApp_chat","Phone_agent","Inperson_visit","CA_setup"], [30,30,20,8,12])[0]

        # ── Section E: Product Preferences ───────────────────────────────
        if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]:
            r["vendor_pay_usefulness"] = weighted_choice([1,2,3,4,5], [2,5,15,38,40])[0]
            r["settlement_pref"] = weighted_choice(["Same_day","Next_day","T+3","Lowest_fee"], [40,30,20,10])[0]
            r["max_fee_tolerance"] = weighted_choice(
                ["<1pct","1-1.5pct","1.5-2pct","2-2.5pct",">2.5pct","No_fee"], [3,10,25,35,22,5])[0]
        elif persona == "P2_REBroker":
            r["vendor_pay_usefulness"] = weighted_choice([1,2,3,4,5], [3,7,20,40,30])[0]
            r["settlement_pref"] = weighted_choice(["Same_day","Next_day","T+3","Lowest_fee"], [30,35,25,10])[0]
            r["max_fee_tolerance"] = weighted_choice(
                ["<1pct","1-1.5pct","1.5-2pct","2-2.5pct",">2.5pct","No_fee"], [5,15,30,30,12,8])[0]
        elif persona == "P3_TradSME":
            r["vendor_pay_usefulness"] = weighted_choice([1,2,3,4,5], [10,20,30,25,15])[0]
            r["settlement_pref"] = weighted_choice(["Same_day","Next_day","T+3","Lowest_fee"], [10,20,30,40])[0]
            r["max_fee_tolerance"] = weighted_choice(
                ["<1pct","1-1.5pct","1.5-2pct","2-2.5pct",">2.5pct","No_fee"], [15,25,25,15,5,15])[0]
        else:  # P5_Skeptic
            r["vendor_pay_usefulness"] = weighted_choice([1,2,3,4,5], [8,15,28,30,19])[0]
            r["settlement_pref"] = weighted_choice(["Same_day","Next_day","T+3","Lowest_fee"], [20,28,28,24])[0]
            r["max_fee_tolerance"] = weighted_choice(
                ["<1pct","1-1.5pct","1.5-2pct","2-2.5pct",">2.5pct","No_fee"], [10,20,28,22,8,12])[0]

        # Feature ratings 1-5
        features = ["feat_reconciliation","feat_analytics","feat_virtual_cards",
                    "feat_gst_filing","feat_invoice_finance","feat_payroll_bridge"]
        if persona in ["P1_TechFounder","P4_Consultant"]:
            base = [4.2, 4.0, 3.8, 4.1, 3.5, 3.7]
        elif persona == "P2_REBroker":
            base = [3.8, 3.5, 3.0, 4.2, 3.8, 3.2]
        elif persona == "P6_PowerUser":
            base = [4.5, 4.3, 4.2, 4.4, 4.1, 4.0]
        elif persona == "P3_TradSME":
            base = [3.2, 2.8, 2.5, 3.5, 3.0, 2.8]
        else:
            base = [3.5, 3.2, 3.0, 3.8, 3.3, 3.1]
        for feat, b in zip(features, base):
            val = int(np.clip(np.round(b + np.random.normal(0, 0.8)), 1, 5))
            r[feat] = val

        r["fx_interest"] = weighted_choice(
            ["Yes_intl_vendors","Maybe_future","No_domestic","Not_applicable"],
            [35,25,30,10] if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]
            else [10,20,55,15])[0]

        # ── Section F: Spending Power ─────────────────────────────────────
        if persona in ["P6_PowerUser"]:
            r["credit_card_limit"] = weighted_choice(["<1L","1L-5L","5L-15L",">15L","No_card"], [1,8,35,55,1])[0]
            r["saas_budget"] = weighted_choice(["Nothing","<500","500-1500","1500-3000",">3000"], [3,7,20,35,35])[0]
        elif persona in ["P1_TechFounder","P4_Consultant"]:
            r["credit_card_limit"] = weighted_choice(["<1L","1L-5L","5L-15L",">15L","No_card"], [5,30,42,18,5])[0]
            r["saas_budget"] = weighted_choice(["Nothing","<500","500-1500","1500-3000",">3000"], [5,15,35,30,15])[0]
        elif persona == "P2_REBroker":
            r["credit_card_limit"] = weighted_choice(["<1L","1L-5L","5L-15L",">15L","No_card"], [8,35,38,12,7])[0]
            r["saas_budget"] = weighted_choice(["Nothing","<500","500-1500","1500-3000",">3000"], [10,20,35,25,10])[0]
        elif persona == "P3_TradSME":
            r["credit_card_limit"] = weighted_choice(["<1L","1L-5L","5L-15L",">15L","No_card"], [20,35,22,5,18])[0]
            r["saas_budget"] = weighted_choice(["Nothing","<500","500-1500","1500-3000",">3000"], [35,30,22,10,3])[0]
        else:
            r["credit_card_limit"] = weighted_choice(["<1L","1L-5L","5L-15L",">15L","No_card"], [10,35,35,12,8])[0]
            r["saas_budget"] = weighted_choice(["Nothing","<500","500-1500","1500-3000",">3000"], [15,25,30,20,10])[0]

        r["gst_importance"] = weighted_choice([1,2,3,4,5],
            [2,5,15,35,43] if persona in ["P2_REBroker","P6_PowerUser"]
            else [5,10,25,35,25])[0]

        r["accounting_software"] = weighted_choice(
            ["Tally","Zoho_QB","Excel","CA_manages","Busy_Marg","No_system"],
            [15,45,22,10,5,3] if persona in ["P1_TechFounder","P4_Consultant"]
            else [40,15,20,20,8,7] if persona in ["P2_REBroker","P3_TradSME"]
            else [25,30,20,15,6,4])[0]

        # ── Section G: Social, Seasonal, Lifestyle ────────────────────────
        saas_tools = ["CRM","Cloud_storage","HR_payroll","Project_mgmt","Accounting_sw","None"]
        if persona in ["P1_TechFounder","P4_Consultant"]:
            st_probs = [0.55,0.80,0.45,0.60,0.70,0.05]
        elif persona == "P6_PowerUser":
            st_probs = [0.60,0.75,0.55,0.50,0.75,0.02]
        elif persona == "P3_TradSME":
            st_probs = [0.10,0.25,0.15,0.08,0.35,0.30]
        else:
            st_probs = [0.30,0.55,0.30,0.30,0.50,0.15]
        for st, p in zip(saas_tools, st_probs):
            r[f"tool_{st}"] = int(np.random.random() < p)

        r["peer_network_size"] = weighted_choice(["None","1-5","6-20","20+"],
            [10,35,38,17] if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]
            else [20,40,30,10])[0]

        r["social_influence_score"] = weighted_choice([1,2,3,4,5],
            [3,8,20,38,31] if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]
            else [8,18,32,28,14])[0]

        r["pain_point"] = weighted_choice(
            ["Cashflow_timing","Manual_reconciliation","Card_not_accepted",
             "High_bank_charges","GST_complex","No_single_view"],
            [30,20,25,8,10,7] if persona in ["P2_REBroker","P6_PowerUser"]
            else [20,25,20,10,15,10])[0]

        r["cash_pressure_months"] = weighted_choice(
            ["Jan-Mar","Apr-Jun","Jul-Sep","Oct-Dec","Consistent","No_pattern"],
            [25,20,15,28,8,4] if persona in ["P2_REBroker"]
            else [20,18,15,20,18,9])[0]

        r["festival_spike"] = weighted_choice(
            ["Yes_significant","Yes_slightly","No_change","Decreases"],
            [30,35,28,7] if persona in ["P2_REBroker","P3_TradSME"]
            else [15,30,45,10])[0]

        # ── Bias controls ─────────────────────────────────────────────────
        r["bc1_fee_tradeoff"] = weighted_choice(["Instant_2.5pct","Classic_1pct"],
            [65,35] if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]
            else [35,65])[0]

        r["bc2_disqualifier"] = weighted_choice(
            ["Yes_need_float","Probably_yes","Maybe","No_only_situational"],
            [35,30,20,15] if persona in ["P1_TechFounder","P4_Consultant","P6_PowerUser"]
            else [20,25,25,30] if persona == "P2_REBroker"
            else [10,18,25,47])[0]

        # ── TARGET: Adoption Intent ───────────────────────────────────────
        # Base probabilities per persona then modulated by key features
        if persona == "P1_TechFounder":
            base_probs = [0.05, 0.18, 0.42, 0.35]
        elif persona == "P2_REBroker":
            base_probs = [0.12, 0.30, 0.35, 0.23]
        elif persona == "P3_TradSME":
            base_probs = [0.38, 0.38, 0.18, 0.06]
        elif persona == "P4_Consultant":
            base_probs = [0.05, 0.15, 0.42, 0.38]
        elif persona == "P5_Skeptic":
            base_probs = [0.40, 0.35, 0.18, 0.07]
        else:  # P6_PowerUser
            base_probs = [0.02, 0.10, 0.35, 0.53]

        probs = list(base_probs)

        # Modulate by working capital crisis
        if r["working_capital_crisis"] in ["Yes_severe","Ongoing"]:
            probs[2] += 0.08; probs[3] += 0.10; probs[0] -= 0.09; probs[1] -= 0.09
        # Modulate by fraud experience
        if r["fraud_experience"] == "Yes_significant":
            probs[0] += 0.12; probs[1] += 0.08; probs[2] -= 0.10; probs[3] -= 0.10
        # Modulate by digital comfort
        if r["digital_comfort"] == "Very_comfortable":
            probs[3] += 0.07; probs[0] -= 0.04; probs[1] -= 0.03
        elif r["digital_comfort"] == "Not_comfortable":
            probs[0] += 0.08; probs[1] += 0.05; probs[2] -= 0.07; probs[3] -= 0.06
        # Modulate by bc2 disqualifier
        if r["bc2_disqualifier"] == "No_only_situational":
            probs[0] += 0.10; probs[1] += 0.05; probs[2] -= 0.08; probs[3] -= 0.07
        # Clip and renormalise
        probs = [max(0.01, p) for p in probs]
        total = sum(probs)
        probs = [p/total for p in probs]

        r["adoption_intent"] = int(np.random.choice([0,1,2,3], p=probs))

        rows.append(r)
    return rows

# ── Build full dataset ────────────────────────────────────────────────────────
all_rows = []
for persona, count in PERSONAS.items():
    all_rows.extend(build_persona(persona, count))

df = pd.DataFrame(all_rows)
df.insert(0, "respondent_id", [f"R{str(i+1).zfill(4)}" for i in range(N)])

# ── Inject noise: 5% random flips on categorical cols ────────────────────────
cat_cols = ["age_group","city_tier","business_type","settlement_pref",
            "digital_comfort","biggest_concern","accounting_software"]
for col in cat_cols:
    mask = np.random.random(N) < 0.05
    vals = df[col].values.copy()
    unique_vals = df[col].unique()
    for i in np.where(mask)[0]:
        others = [v for v in unique_vals if v != vals[i]]
        if others:
            vals[i] = random.choice(others)
    df[col] = vals

# ── Inject missing values: 3% on sensitive cols ──────────────────────────────
missing_cols = ["peer_network_size","biggest_concern","fraud_experience","rbi_importance"]
for col in missing_cols:
    mask = np.random.random(N) < 0.03
    df.loc[mask, col] = np.nan

# ── Inject outliers: 40 extreme rows ─────────────────────────────────────────
outlier_idx = np.random.choice(N, 40, replace=False)
df.loc[outlier_idx[:20], "monthly_payment_vol"] = ">25L"
df.loc[outlier_idx[:20], "credit_card_limit"] = ">15L"
df.loc[outlier_idx[:20], "city_tier"] = "Tier3"  # extreme combo
df.loc[outlier_idx[20:], "max_fee_tolerance"] = "No_fee"
df.loc[outlier_idx[20:], "monthly_payment_vol"] = ">25L"  # contradictory
df.loc[outlier_idx[20:], "adoption_intent"] = np.random.choice([0,3], 20)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "/home/claude/bharatnxt/bharatnxt_survey_data.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved: {out_path}")
print(f"Shape: {df.shape}")
print(f"\nAdoption intent distribution:")
print(df["adoption_intent"].value_counts().sort_index())
print(f"\nPersona distribution:")
print(df["persona_label"].value_counts())
