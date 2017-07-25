import pandas as pd

df = pd.read_table('C:\\Test_Data\\smsspamcollection\\SMSSpamCollection',
				   sep='\t',
				   header=None,
				   names=['label', 'sms_message'])

print(df.head())