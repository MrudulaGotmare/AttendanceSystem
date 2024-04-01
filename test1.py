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
import pandas as pd

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
        
        # Initialize defaultdict to store attendance data
        attendance_dict = defaultdict(dict)
        for name, attendancedate in data:
            attendance_dict[attendancedate][name] = "Present"  # Assuming all retrieved records represent attendance
        
        # Calculate the total number of days each student attended and total attendance days for each student
        student_total_days = defaultdict(int)
        student_present_days = defaultdict(int)
        for attendancedate, attendance_data in attendance_dict.items():
            for name, status in attendance_data.items():
                student_total_days[name] += 1
                if status == "Present":
                    student_present_days[name] += 1
        
        # Calculate percentage for each student based on total days available
        student_percentage = {}
        for name, total_days in student_total_days.items():
            present_days = student_present_days[name]
            student_percentage[name] = (present_days / total_days) * 100 if total_days != 0 else 0
        
        # Update Google Sheets
        sheet = client.open_by_key(sheets_id)
        worksheet = sheet.get_worksheet(0)  # Assuming the data will be inserted into the first worksheet
        worksheet.clear()
        
        # Insert headers
        headers = ['Date'] + sorted({name for date_data in attendance_dict.values() for name in date_data.keys()})
        worksheet.insert_row(headers, index=1)
        
        # Insert data rows
        row_index = 2  # Start from the second row (after headers)
        for attendancedate, attendance_data in sorted(attendance_dict.items()):
            formatted_date = datetime.strptime(attendancedate, "%Y-%m-%d").strftime("%Y-%m-%d")
            row_data = [formatted_date] + [attendance_data.get(name, "Absent") for name in headers[1:]]
            worksheet.insert_row(row_data, index=row_index)
            row_index += 1
        
        # Insert percentage row
        percentage_row = ["Percentage"] + [student_percentage.get(name, 0) for name in headers[1:]]
        worksheet.insert_row(percentage_row, index=row_index)
        
        # Show success message
        QMessageBox.information(self, "Attendance System", "Data has been successfully updated in Google Sheets.")
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


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()