----

Giving access to s3

1. Grab the policy file: aws s3api  get-bucket-acl --bucket modis-l2 --endpoint https://s3.nautilus.optiputer.net 

2. Add Grantee's in JSON file

3. Put it back: aws s3api put-bucket-policy --bucket modis-l2 --policy s3_modis-l2_policy.json --endpoint https://s3.nautilus.optiputer.net 