cd code
mkdir e2e_loan_approval_predictor
python --version

python -m venv venv

Get-ExecutionPolicy
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\venv\Scripts\Activate.ps1

pip freeze > requirements.txt

mkdir app

pip install fastapi uvicorn
pip install pandas joblib sk-scikit-learn
pip list

uvicorn app.main:app --reload