# Intent Classification Evaluation

- Method: **rules**
- Holdout size: 640
- Accuracy: **1.0000**
- Macro F1: **1.0000**

## Classification Report
```
                          precision    recall  f1-score   support

    comparative_analysis       1.00      1.00      1.00       124
             forecasting       1.00      1.00      1.00       125
operational_optimization       1.00      1.00      1.00       127
  performance_monitoring       1.00      1.00      1.00       264

                accuracy                           1.00       640
               macro avg       1.00      1.00      1.00       640
            weighted avg       1.00      1.00      1.00       640

```

## Confusion Matrix
Labels: ['comparative_analysis', 'forecasting', 'operational_optimization', 'performance_monitoring']
```
[[124   0   0   0]
 [  0 125   0   0]
 [  0   0 127   0]
 [  0   0   0 264]]
```