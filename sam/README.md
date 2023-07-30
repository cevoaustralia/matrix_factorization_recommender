# Get Products Lambda
To deploy the SAM-based Lambda, login with your SSO credentials with the following command (yours will be different):

`aws sso login --sso-session mfrec`

Then deploy the SAM-based Lambda with the following command (your profile name will be different too):

`sam build && sam deploy --region ap-southeast-2 --profile Cevo-Dev.AWSFullAccountAdmin --no-confirm-changeset --parameter-overrides 'ParameterKey=BestModel,ParameterValue=mf-recommender-2023-07-09-12-25-43.pkl'`

