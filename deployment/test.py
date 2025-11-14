from app import load_model
test_data = {'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 1958.928726,
            'total_debt_outstanding': 8228.75252,
            'income': 26648.43525 ,
            'years_employed' : 2,
            'fico_score' : 572,
            'debt_ratio' : 0.30878933201152964}

def test_predict():
    prediction = load_model(test_data)
    assert prediction == 1, "incorrect prediction"
