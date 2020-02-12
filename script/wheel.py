import boto3

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(name="pytorch-scatter")
objects = bucket.objects.all()
wheels = sorted([obj.key[4:] for obj in objects if obj.key[:3] == 'whl'])
print(wheels)

# <!DOCTYPE html>
# <html>
#   <body>
#     <a href="/frob/">frob</a>
#     <a href="/spamspamspam/">spamspamspam</a>
#   </body>
# </html>

url = 'https://pytorch-scatter.s3.eu-central-1.amazonaws.com/whl/{}.whl'
