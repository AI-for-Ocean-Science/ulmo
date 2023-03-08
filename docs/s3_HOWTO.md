----

Giving access to s3

Nautilus notes: https://ucsd-prp.gitlab.io/userdocs/storage/ceph-s3/#ceph_s3_3 

1. Grab the policy file: aws s3api  get-bucket-acl --bucket modis-l2 --endpoint https://s3-west.nrp-nautilus.io 

1. Add Grantee, keeping all previous: aws s3api put-bucket-acl --profile default --bucket modis-l2 --grant-full-control id=profx,id=erdong,id=petercornillon,id=mskelm,id=aagabin --endpoint https://s3-west.nrp-nautilus.io

1. Add Grantee, keeping all previous: aws s3api put-bucket-acl --profile default --bucket llc --grant-full-control id=profx,id=erdong,id=petercornillon,id=aagabin --endpoint https://s3-west.nrp-nautilus.io

1. Add Grantee, keeping all previous: aws s3api put-bucket-acl --profile default --bucket viirs --grant-full-control id=profx,id=erdong,id=petercornillon,id=aagabin --endpoint https://s3-west.nrp-nautilus.io

   Note:  the id may not exactly match that on Matrix.  e.g. katmar instead of katmar4141

1. Create policy.json file.  See examples in ulmo/nautilus

1. Modify policy: aws --endpoint-url https://s3-west.nrp-nautilus.io s3api put-bucket-policy --bucket modis-l2 --policy file://s3_modis-l2_policy.json

1. Check:  aws --endpoint https://s3-west.nrp-nautilus.io s3api get-bucket-policy --bucket modis-l2 --query Policy --output text