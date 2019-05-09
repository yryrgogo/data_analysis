from flask import Flask, jsonify, request

from cashman.model.expense import Expense, ExpenseSchema
from cashman.model.income import Income, IncomeSchema
from cashman.model.transaction_type import TransactionType


app = Flask(__name__)


transactions = [
    Income('Salary', 5000),
    Income('Dividends', 200),
    Expense('pizza', 50),
    Expense('Rock Concert', 100)
]


@app.route('/incomes'
def get_incomes():
    """
    1. manyってなに？
    2. filterの挙動を忘れた
    """
    schema = IncomeSchema(many=True)
    incomes = schema.dump(
        filter(lambda t: t.type == TransactionType.INCOME, transactions )
    )

    return jsonify(incomes.data)


@app.route('/incomes', methods=['POST'])
def add_income():
    """
    1. POSTは待ちの姿勢のmethodsか？それとも新しいデータを追加する為のmethodsか？
    2. request.get_json()で新しい入力データを読み込むのか？
    3. 204ってなに？なんで空文字を返すの？
    """
    income = IncomeSchema().load(request.get_json())
    transactions.append(income.data)
    return "", 204


@app.route('/expenses')
def get_expenses():
    schema = ExpenseSchema(many=True)
    expenses = schema.dump(
        filter(lambda t: t.type == TransactionType.EXPENSE, transactions)
    )
    return jsonify(expenses.data)


@app.route('/expenses', methods=['POST'])
def add_expense():
    expense = ExpenseSchema().load(request.get_json())
    transactions.append(expense.data)
    return "", 204


if __name__=="__main__":
    app.run()
