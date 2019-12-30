from demo_final import demo_flask
image_file="/home/hu/Common/ChineseAddress_OCR/upload/2.png"
output_file,ret_total = demo_flask(image_file)
print('Recongition Result:')
print(ret_total)
