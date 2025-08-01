from twilio.rest import Client
accountSid ="ACf5cd38f4b4574fa74033982960252181"
authToken ="d0e8dba5aa5feac93757fa5d6df31a87"

client = Client(accountSid,authToken)

try:
    message = client.messages.create(
        body="元宵節快樂",
        from_='+16288775303',
        to='+886906787125'
    )
    print("Message SID:", message.sid)

except Exception as e:
    print("錯誤：", str(e))