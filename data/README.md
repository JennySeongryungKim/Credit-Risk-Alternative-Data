</h2><b>Overview</b></h2>

This project uses the <b>Home Credit Default Risk</b> dataset, which contains loan application data and various behavioral/historical information about clients.
The goal is to predict whether a client will have payment difficulties.

---

</h3><b>Source</b></h3>

This study is based on the <b>“Home Credit Default Risk”</b> dataset from <a href="https://www.kaggle.com/competitions/home-credit-default-risk">Kaggle</a>, which aims to predict the credit repayment capability of loan applicants.
The dataset contains comprehensive information from <b>Home Credit Group</b>, including traditional credit records (<code>application</code>, <code>bureau</code>, <code>payment</code>, <code>POS</code>, <code>card</code>, <code>installment</code>) and alternative behavioral features.
It was originally published by <b>Anna Montoya, inversion, Kirill Odintsov, and Martin Kotek (2018)</b> on Kaggle.

---

</h2><b>Data Structure</b></h2>

The dataset consists of <b>7 interconnected tables:</b>

<pre>
application_{train|test}.csv  (Main table)
├── bureau.csv                (Credit Bureau data)
│   └── bureau_balance.csv    (Monthly balance history)
├── previous_application.csv  (Previous Home Credit loans)
│   ├── POS_CASH_balance.csv  (Monthly balance - POS/Cash loans)
│   ├── credit_card_balance.csv (Monthly balance - Credit cards)
│   └── installments_payments.csv (Payment history)
</pre>

---

</h3><b>Table Descriptions</b></h3>

</br>

</b>1. application_{train|test}.csv (Main Table)</b>

1. <b>Purpose:</b> Primary loan application data with client information
2. <b>Granularity:</b> One row per loan application (<code>SK_ID_CURR</code>)
3. <b>Key Information:</b>

* Target variable (binary): Payment difficulties indicator
* Loan details (amount, annuity, goods price)
* Client demographics and income
* External credit scores (<code>EXT_SOURCE_1/2/3</code>)

</br>

</b>2. bureau.csv</b>

1. <b>Purpose:</b> Credit history from other financial institutions
2. <b>Granularity:</b> One row per external loan (<code>SK_ID_BUREAU</code>)
3. <b>Relationship:</b> Multiple loans per client (<code>SK_ID_CURR</code>)
4. <b>Key Information:</b>

* Loan types, amounts, and status from Credit Bureau
* Active/closed loan indicators

</br>

</b>3. bureau_balance.csv</b>

1. <b>Purpose:</b> Monthly balance snapshot of bureau credits
2. <b>Granularity:</b> One row per month per bureau loan
3. <b>Relationship:</b> Links to <code>bureau.csv</code> via <code>SK_ID_BUREAU</code>
4. <b>Key Information:</b> Monthly status and balance trends

</br>

</b>4. previous_application.csv</b>

1. <b>Purpose:</b> Historical Home Credit loan applications
2. <b>Granularity:</b> One row per previous application (<code>SK_ID_PREV</code>)
3. <b>Relationship:</b> Multiple previous applications per client
4. <b>Key Information:</b>

* Previous loan parameters
* Application status (approved / refused / cancelled)

</br>

</b>5. POS_CASH_balance.csv</b>

1. <b>Purpose:</b> Monthly balance of POS and cash loans
2. <b>Granularity:</b> One row per month per previous loan
3. <b>Relationship:</b> Links to <code>previous_application.csv</code> via <code>SK_ID_PREV</code>

</br>

</b>6. credit_card_balance.csv</b>

1. <b>Purpose:</b> Monthly credit card balance from Home Credit
2. <b>Granularity:</b> One row per month per previous credit card
3. <b>Relationship:</b> Links to <code>previous_application.csv</code> via <code>SK_ID_PREV</code>

</br>

</b>7. installments_payments.csv</b>

1. <b>Purpose:</b> Payment history for previous loans
2. <b>Granularity:</b> One row per payment installment
3. <b>Relationship:</b> Links to <code>previous_application.csv</code> via <code>SK_ID_PREV</code>
4. <b>Key Information:</b> Expected vs actual payment amounts and dates

---

