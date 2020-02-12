import boto3

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(name="pytorch-scatter")
objects = bucket.objects.all()
wheels = sorted([obj.key[4:] for obj in objects if obj.key[-3:] == 'whl'])

content = '<!DOCTYPE html>\n<html>\n<body>\n{}\n</body>\n</html>'
url = 'https://pytorch-scatter.s3.eu-central-1.amazonaws.com/whl/{}'
links = ['<a href="{}">{}</a><br/>'.format(url.format(w), w) for w in wheels]
content = content.format('\n'.join(links))

with open('index.html', 'w') as f:
    f.write(content)

bucket.Object('whl/index.html').upload_file(
    Filename='index.html', ExtraArgs={
        'ContentType': 'text/html',
        'ACL': 'public-read'
    })
