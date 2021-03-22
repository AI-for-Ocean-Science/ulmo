----

Giving access to s3

1. Grab the policy file: aws s3api  get-bucket-acl --bucket modis-l2 --endpoint https://s3.nautilus.optiputer.net 

1. Add Grantee, keeping all previous: aws s3api put-bucket-acl --profile default --bucket modis-l2 --grant-full-control id=profx,id=erdong --endpoint https://s3.nautilus.optiputer.net

1. Modify policy -- aws s3api put-bucket-policy --bucket modis-l2 --policy s3_modis-l2_policy.json --endpoint https://s3.nautilus.optiputer.net 