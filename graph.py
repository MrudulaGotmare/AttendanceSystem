from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
from datetime import datetime
import cv2, os, numpy
from PIL import Image
import pandas as pd
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
import gspread
from google.oauth2.service_account import Credentials
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

scopes = [
    "https://www.googleapis.com/auth/spreadsheets"
]

creds = Credentials.from_service_account_file("credential.json", scopes=scopes)
client = gspread.authorize(creds)

sheets_id = "1MNNZbGUQ-Re10s7KHrckP_PW3Vj5z9DBcfFJnHh86l8"
sheet = client.open_by_key(sheets_id)

ui,_=loadUiType('face-reco.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0) #The default screen would be the login screen whose index is 0
        self.LOGINBUTTON.clicked.connect(self.login)
        self.LOGOUTBUTTON.clicked.connect(self.logout)
        self.CLOSEBUTTON.clicked.connect(self.close_window)
        self.TRAINLINK1.clicked.connect(self.training_form)
        self.ATTLINK1.clicked.connect(self.attendance_entry_form)
        self.REPORTSLINK1.clicked.connect(self.reports_form)
        self.TRAININGBACK.clicked.connect(self.show_mainform)
        self.ATTENDANCEBACK.clicked.connect(self.show_mainform)
        self.REPORTSBACK.clicked.connect(self.show_mainform)
        self.TRAININGBUTTON.clicked.connect(self.start_training)
        self.RECORD.clicked.connect(self.record_attendance)
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.show_selected_date_reports)
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")
        self.REPORTS_LINK.clicked.connect(self.sheet_reports)
        self.generateGraphsButton.clicked.connect(self.plot_dummy_data_graphs)

        try:
            con = sqlite3.connect("face-reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER, name TEXT, attendancedate TEXT)")
            con.commit()
            print("Table created successfully")
        except:
            print("Error in database")

    ### LOGIN PROCESS ###    
    def login(self):
        pw = self.PASSWORD.text()
        if(pw=="123"):
            self.PASSWORD.setText("")
            self.LOGININFO.setText("")
            self.tabWidget.setCurrentIndex(1)
        else:
            self.LOGININFO.setText("Invalid Password..")
            self.PASSWORD.setText("")

    ### LOGOUT PROCESS ###    
    def logout(self):
        self.tabWidget.setCurrentIndex(0)

    ### SHOW TRAINING FORM ###    
    def training_form(self):
        self.tabWidget.setCurrentIndex(2)

    ### SHOW ATTENDANCE ENTRY FORM ###    
    def attendance_entry_form(self):
        self.tabWidget.setCurrentIndex(3)

    ### SHOW REPORT FORM ###
    def reports_form(self):
        self.tabWidget.setCurrentIndex(4)
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance")
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTS.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number,column_number,QTableWidgetItem(str(data)))
        self.REPORTS.setHorizontalHeaderLabels(['Id','Name','Date'])
        self.REPORTS.setColumnWidth(0,20)
        self.REPORTS.setColumnWidth(1,70)
        self.REPORTS.setColumnWidth(2,80)
        self.REPORTS.verticalHeader().setVisible(False)

    ### SHOW SELECTED DATE REPORTS ###
    def show_selected_date_reports(self):
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '" + str(self.dateEdit.date().toPyDate()) + "'")
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTS.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number,column_number,QTableWidgetItem(str(data)))
        self.REPORTS.setHorizontalHeaderLabels(['Id','Name','Date'])
        self.REPORTS.setColumnWidth(0,10)
        self.REPORTS.setColumnWidth(1,60)
        self.REPORTS.setColumnWidth(2,70)
        self.REPORTS.verticalHeader().setVisible(False)

    ### SHOW REPORT IN GOOGLE SHEETS
    def sheet_reports(self):
        con = sqlite3.connect("face-reco.db")
        cursor = con.cursor()
        cursor.execute("SELECT name, attendancedate FROM attendance")
        data = cursor.fetchall()
        con.close()

    # Organize the data into a dictionary with attendancedate as keys and names as values
        attendance_dict = defaultdict(dict)
        for name, attendancedate in data:
            attendance_dict[attendancedate][name] = "Present"  # Assuming all retrieved records represent attendance

    # Open the Google Sheets document
        sheet = client.open_by_key(sheets_id)
        worksheet = sheet.get_worksheet(0)  # Assuming the data will be inserted into the first worksheet
        worksheet.clear()

    # Write the headers (Date and Student names)
        headers = ['Date'] + sorted({name for date_data in attendance_dict.values() for name in date_data.keys()})
        worksheet.insert_row(headers, index=1)

    # Write the attendance data for each date
        row_index = 2  # Start from the second row (after headers)
        for attendancedate, attendance_data in sorted(attendance_dict.items()):
        # Convert the date to a string in the desired format
            formatted_date = datetime.strptime(attendancedate, "%Y-%m-%d").strftime("%Y-%m-%d")
        # Prepare the row data
            row_data = [formatted_date] + [attendance_data.get(name, "Absent") for name in headers[1:]]
        # Update the row in the worksheet
            worksheet.insert_row(row_data, index=row_index)
            row_index += 1  # Move to the next row

    # Inform the user that the data has been successfully updated in Google Sheets
        QMessageBox.information(self, "Attendance System", "Data has been successfully updated in Google Sheets.")

    # Open the URL of the Google Sheets document
        url = QUrl("https://docs.google.com/spreadsheets/d/" + sheets_id)
        QDesktopServices.openUrl(url)


    ### SHOW MAIN FORM ###    
    def show_mainform(self):
        self.tabWidget.setCurrentIndex(1)

    ### CLOSE WINDOW PROCESS ###    
    def close_window(self):
        self.close()

    ### TRAINING PROCESS ###
    def start_training(self):
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = self.traineeName.text()
        path = os.path.join(datasets,sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            (width,height) = (130,100)
            face_cascade = cv2.CascadeClassifier(haar_file)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.trainingCount.text()) +1:
                print(count)
                (_,im) = webcam.read()
                ### converting image to grayscale ###
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,4)
                ### x,y is the image starting position ###
                for (x,y,w,h) in faces:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                    face = gray[y:y+h,x:x+w]
                    face_resize = cv2.resize(face,(width,height))
                    cv2.imwrite('%s/%s.png'%(path,count),face_resize)
                count += 1
                cv2.imshow('OpenCV',im)
                key = cv2.waitKey(10)
                if key == 27:
                    break
            webcam.release()
            cv2.destroyAllWindows()
            path=""
            QMessageBox.information(self,"Attendance System","Training completed Successfully")
            self.traineeName.setText("")
            self.trainingCount.setText("100")
    
    ### RECORD ATTENDANCE ###    
    def record_attendance(self):
        self.currentprocess.setText("Process started.. waiting..")
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        datasets = 'datasets'
        (images,labels,names,id) = ([],[],{},0)
        for(subdirs,dirs,files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets,subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + "/" + filename
                    label = id
                    images.append(cv2.imread(path,0))
                    labels.append(int(label))
                id += 1
        (images,labels) = [numpy.array(lis) for lis in [images,labels]]
        print(images,labels)
        (width, height) = (130,100)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images,labels)
        webcam = cv2.VideoCapture(0)
        cnt=0
        while True:
            (_,im) = webcam.read()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
                face = gray[y:y+h,x:x+w]
                face_resize = cv2.resize(face,(width,height))
                prediction = model.predict(face_resize)
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                if(prediction[1]<800):
                    cv2.putText(im,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                    print(names[prediction[0]])
                    self.currentprocess.setText("Dectected face" + names[prediction[0]])
                    attendanceid = 0
                    available = False
                    try:
                        connection = sqlite3.connect("face-reco.db")
                        cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                        result = cursor.fetchall()
                        if result:
                            for maxid in result:
                                attendanceid = int(maxid[0]+1)
                    except:
                        attendanceid = 1
                    print(attendanceid)

                    try:
                        con = sqlite3.connect("face-reco.db")
                        cursor = con.execute("SELECT * FROM attendance WHERE name='" + str(names[prediction[0]]) + "' and attendancedate = '" + str(date.today()) + "'")
                        result = cursor.fetchall()
                        if result:
                            available=True
                        if(available==False):
                            con.execute("INSERT INTO attendance VALUES("+ str(attendanceid) +",'"+ str(names[prediction[0]]) +"','"+ str(date.today()) +"') ")
                            con.commit()
                    except:
                        print("ERROR IN DATABASE INSERT")
                    print("Attendance Registered successfully")
                    self.currentprocess.setText("Attendance entered for " + names[prediction[0]])
                    cnt=0
                else:
                    cnt+=1
                    cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
                    if(cnt>100):
                        print("Unknown person")
                        self.currentprocess.setText("Unknown Person")
                        cv2.imwrite('unknown.jpg',im)
                        cnt=0
            cv2.imshow("Face Recognition",im)
            key = cv2.waitKey(10)
            if key==27:
                break
        webcam.release()
        cv2.destroyAllWindows()
        
    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def plot_dummy_data_graphs(self):
        # Generate dummy data
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train an SVM model
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)

        # Plot the confusion matrix
        self.plot_confusion_matrix(y_test, svm_model.predict(X_test), classes=np.unique(y), title='Confusion Matrix')

        # Plot SVM decision boundary
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = svm_model.decision_function(xy).reshape(XX.shape)

        # Plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

        plt.title('SVM Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        # Show the plots
        plt.show()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
