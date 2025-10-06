</h2></b>Overview</b></h2>
This project uses the Home Credit Default Risk dataset, which contains loan application data and various behavioral/historical information about clients. The goal is to predict whether a client will have payment difficulties.

</h3></b>Source</b></h3>
This study is based on the “Home Credit Default Risk” dataset from Kaggle, which aims to predict the credit repayment capability of loan applicants. The dataset contains comprehensive information from Home Credit Group, including traditional credit records (application, bureau, payment, POS, card, installment) and alternative behavioral features. It was originally published by Anna Montoya, inversion, Kirill Odintsov, and Martin Kotek (2018) on Kaggle.


</h2></b>Data Structure</b></h2>
The dataset consists of 7 interconnected tables:
application_{train|test}.csv (Main table)
├── bureau.csv (Credit Bureau data)
│   └── bureau_balance.csv (Monthly balance history)
├── previous_application.csv (Previous Home Credit loans)
│   ├── POS_CASH_balance.csv (Monthly balance - POS/Cash loans)
│   ├── credit_card_balance.csv (Monthly balance - Credit cards)
│   └── installments_payments.csv (Payment history)


</h3></b>Table Descriptions</b></h3>

</b>1. application_{train|test}.csv (Main Table)</b>
1) Purpose: Primary loan application data with client information
2) Granularity: One row per loan application (SK_ID_CURR)
3) Key Information:
- Target variable (binary): Payment difficulties indicator
- Loan details (amount, annuity, goods price)
- Client demographics and income
- External credit scores (EXT_SOURCE_1/2/3)


</b>2. bureau.csv</b>
1) Purpose: Credit history from other financial institutions
2) Granularity: One row per external loan (SK_ID_BUREAU)
3) Relationship: Multiple loans per client (SK_ID_CURR)
4) Key Information:
- Loan types, amounts, and status from Credit Bureau
- Active/closed loan indicators


</b>3. bureau_balance.csv</b>
1) Purpose: Monthly balance snapshot of bureau credits
2) Granularity: One row per month per bureau loan
3) Relationship: Links to bureau.csv via SK_ID_BUREAU
4) Key Information: Monthly status and balance trends

</b>4. previous_application.csv</b>
1) Purpose: Historical Home Credit loan applications
2) Granularity: One row per previous application (SK_ID_PREV)
3) Relationship: Multiple previous applications per client
4) Key Information:
- Previous loan parameters
- Application status (approved/refused/cancelled)


</b>5. POS_CASH_balance.csv</b>
1) Purpose: Monthly balance of POS and cash loans
2) Granularity: One row per month per previous loan
3) Relationship: Links to previous_application.csv via SK_ID_PREV

</b>6. credit_card_balance.csv</b>
1) Purpose: Monthly credit card balance from Home Credit
2) Granularity: One row per month per previous credit card
3) Relationship: Links to previous_application.csv via SK_ID_PREV

</b>7. installments_payments.csv</b>
1) Purpose: Payment history for previous loans
2) Granularity: One row per payment installment
3) Relationship: Links to previous_application.csv via SK_ID_PREV
4) Key Information: Expected vs actual payment amounts and dates
