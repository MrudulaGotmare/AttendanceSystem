import streamlit as st

# Function to display the login page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Dummy authentication, replace this with your actual authentication logic
        if username == "admin" and password == "password":
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid username or password")
            return False

# Function to display the page after successful login
def after_login_page():
    st.title("Welcome")
    st.write("You are now logged in.")
    st.write("Train User")
    st.text_input("Enter Name", key="User")
    st.number_input("Enter Number", key="123")
    if st.button("Train"):
        name = st.session_state.name
        number = st.session_state.number
        st.write(f"User trained with name: {name} and number: {number}")
    if st.button("Go Back"):
        st.experimental_rerun()

# Function to display the attendance report page
def attendance_report_page():
    st.title("Attendance")
    st.button("Record Attendance")
    # Placeholder for attendance report table
    st.write("Attendance report will be displayed here.")

# Main function to control navigation between pages
def main():
    page = st.sidebar.radio("Navigation", ["Login", "Train User", "Attendance"])
    if page == "Login":
        if login_page():
            st.experimental_rerun()
    elif page == "Train User":
        after_login_page()
    elif page == "Attendance":
        attendance_report_page()

if __name__ == "__main__":
    main()
