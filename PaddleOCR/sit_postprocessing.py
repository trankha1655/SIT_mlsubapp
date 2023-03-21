

class SITPostProcessing:
    def __init__(self) -> None:
        pass

    def __call__(self, res):
        for x in res:
            text = x['transcription']
            if 'IMEI2' in text:
                continue

            if 'IMEI' in text:
                imei_id = text.split('IMEI')[-1]
                imei_id = imei_id.strip().replace('\n', '')

                print("Result: '%s'" % imei_id)
                if len(imei_id) >5:
                    return imei_id
                else:
                    return False