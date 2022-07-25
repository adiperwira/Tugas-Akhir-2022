import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class Classification:
    def __init__(self) -> None:
        self.classification_model = joblib.load('model/classification/model.pkl')
        self.scrapped_news = pd.read_csv('result/tagging/severity_count.csv')
    
    def getClassificationValue(self):
        try:
            document_result = []
            
            for i in range(0, self.scrapped_news.shape[0]):
                title = self.scrapped_news.iloc[i, 0]
                berita = self.scrapped_news.iloc[i, 1]
                time = self.scrapped_news.iloc[i, 2]
                what = self.scrapped_news.iloc[i, 3]
                tanggal_asli = self.scrapped_news.iloc[i, 4]
                orang = self.scrapped_news.iloc[i, 5]
                provinsi = self.scrapped_news.iloc[i, 6]
                kabupaten = self.scrapped_news.iloc[i, 7]
                kecamatan = self.scrapped_news.iloc[i, 8]

                """
                Bagian ini adalah indikator yang perlu disesuaikan untuk masing - masing topik
                """
                mati = self.scrapped_news.iloc[i, 9] 
                luka = self.scrapped_news.iloc[i, 10] 
                kerugianBarang = self.scrapped_news.iloc[i, 11] 
                kerugianUang = self.scrapped_news.iloc[i, 12]
                pemerkosaan = self.scrapped_news.iloc[i, 13]

                keparahan = self.classification_model.predict([[mati,luka, kerugianBarang, kerugianUang, pemerkosaan]])
                document_result.append([title, berita, time, what, tanggal_asli, orang, provinsi, kabupaten, kecamatan, mati, luka, kerugianBarang, kerugianUang, pemerkosaan, keparahan[0]])

            writer = pd.DataFrame(document_result, columns=[
                                'title', 'description', 'time', 'what', 'when', 'who', 'provinsi', 'kabupaten / kota', 'kecamatan', 'mati', 'luka', 'kerugianBarang', 'kerugianUang', 'pemerkosaan','klasifikasi'], index=None)
            writer.to_csv('result/classification_res/result_final.csv', index=False, sep=',')
            return "success"
        except:
            return "error"
        