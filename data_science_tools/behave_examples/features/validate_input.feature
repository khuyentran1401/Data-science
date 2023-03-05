Feature: Input Validation for Customer Churn Prediction

  As a data scientist
  I want to ensure that the input data for my model is valid
  So that I can avoid errors and unexpected behavior in my model

  Scenario: Marital status is incorrect
    Given the value of marital_status in at least one sample is incorrect 
        | customer_id | gender | age | marital_status | education    | occupation   | income | account_balance | credit_score | 
        | 1           | Male | 45  | Married | High School  | Blue Collar  | 50000  | 1000            | 600          | 
        | 2           | Female | 32  | Divorced | Graduate     | White Collar | 70000  | 4000            | 700          | 
        | 3           | Male   | 22  | Single | High School | Student      | 20000  | 300             | 500          | 
        | 4           | Female | 58  | Married | Graduate     | White Collar | 90000  | 12000           | 800          |
        | 5           | Male   | 39  | Married | High School  | Blue Collar  | 60000  | 1500            | 550          |
        | 6           | Female | 28  | Single | High School | Student      | 25000  | 200             | 400          |

    When the input is validated 
    Then an error message should be returned    