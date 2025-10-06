# -*- coding: utf-8 -*-
import os
from src.pipeline.credit_pipeline import CreditRiskPipeline
from src.utils.paths import data_path

if __name__ == "__main__":
    # Define data paths
    train_paths = {
        'application': 'application_train.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'previous_application': 'previous_application.csv',
        'credit_card_balance': 'credit_card_balance.csv',
        'pos_cash_balance': 'POS_CASH_balance.csv',
        'installments_payments': 'installments_payments.csv'
    }

    # Define test data paths
    test_paths = {
        'application': 'application_test.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'previous_application': 'previous_application.csv',
        'credit_card_balance': 'credit_card_balance.csv',
        'pos_cash_balance': 'POS_CASH_balance.csv',
        'installments_payments': 'installments_payments.csv'
    }

    # Run pipeline
    try:
        pipeline = CreditRiskPipeline()
        results = pipeline.run(train_paths, test_paths)

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  📁 preprocessed_data.pkl - Preprocessed datasets")
        print("  📁 preprocessor.pkl - Preprocessor object")
        print("  📊 model_results.csv - All model performance metrics")
        print("  📁 models/ - Saved model files")
        print("  📈 model_comparison.png - Performance visualization")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
