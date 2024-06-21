# Import necessary modules
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random

# Dummy database for user data (can be replaced with a real database)
user_database = {
    "admin": {"password": "admin", "is_admin": True, "bank_name": "", "account_number": "", "balance": 0.0, "remaining_units": 0, "threshold": 0},
    "user1": {"password": "password1", "is_admin": False, "bank_name": "ABSA", "account_number": "1234567891", "balance": 1000.0, "remaining_units": 50, "threshold": 10},
    "user2": {"password": "password2", "is_admin": False, "bank_name": "FNB", "account_number": "1234567892", "balance": 1200.0, "remaining_units": 60, "threshold": 8},
    "user3": {"password": "password3", "is_admin": False, "bank_name": "NEDBANK", "account_number": "1234567893", "balance": 800.0, "remaining_units": 40, "threshold": 12},
    "user4": {"password": "password4", "is_admin": False, "bank_name": "STANDARD BANK", "account_number": "1234567894", "balance": 1500.0, "remaining_units": 70, "threshold": 15},
    "user5": {"password": "password5", "is_admin": False, "bank_name": "CAPITEC", "account_number": "1234567895", "balance": 1100.0, "remaining_units": 55, "threshold": 9},
    "user6": {"password": "password6", "is_admin": False, "bank_name": "TYM BANK", "account_number": "1234567896", "balance": 900.0, "remaining_units": 45, "threshold": 11},
    "user7": {"password": "password7", "is_admin": False, "bank_name": "ABSA", "account_number": "1234567897", "balance": 1300.0, "remaining_units": 65, "threshold": 10},
    "user8": {"password": "password8", "is_admin": False, "bank_name": "FNB", "account_number": "1234567898", "balance": 950.0, "remaining_units": 47, "threshold": 8},
    "user9": {"password": "password9", "is_admin": False, "bank_name": "NEDBANK", "account_number": "1234567899", "balance": 1150.0, "remaining_units": 58, "threshold": 12},
    "user10": {"password": "password10", "is_admin": False, "bank_name": "STANDARD BANK", "account_number": "1234567800", "balance": 1400.0, "remaining_units": 72, "threshold": 15}
}

# Define a class for the electricity top-up app
class ElectricityTopUpApp:
    def __init__(self, user):
        self.user = user
        user_data = user_database[user]
        self.bank_name = user_data["bank_name"]
        self.account_number = user_data["account_number"]
        self.balance = user_data["balance"]
        self.remaining_units = user_data["remaining_units"]
        self.threshold = user_data["threshold"]
        self.unit_price = 1.20  # Set unit price as R1.20 per unit

    def top_up(self, units, meter_number):
        # Calculate total amount
        total_amount = units * self.unit_price

        # Deduct amount from bank balance (simulated)
        if self.balance >= total_amount:
            self.balance -= total_amount
            self.remaining_units += units
            st.success(f"Successfully topped up {units} units to meter number {meter_number}. Remaining units: {self.remaining_units}")
        else:
            st.error("Insufficient balance for top-up.")

    def consume_units(self, units):
        if units <= self.remaining_units:
            self.remaining_units -= units
            st.info(f"Consumed {units} units. Remaining units: {self.remaining_units}")
        else:
            st.error("Insufficient units for consumption.")

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_consumption_data(self):
        # Dummy data for visualization
        weekly_data = {"Date": [], "Units": [], "Cost": []}
        monthly_data = {"Date": [], "Units": [], "Cost": []}
        # Fill dummy data (can be replaced with actual data)
        for i in range(1, 8):
            weekly_data["Date"].append((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"))
            weekly_data["Units"].append(i * 10)
            weekly_data["Cost"].append(round(i * 10 * self.unit_price, 2))
        for i in range(1, 31):
            monthly_data["Date"].append((datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"))
            monthly_data["Units"].append(i * 5)
            monthly_data["Cost"].append(round(i * 5 * self.unit_price, 2))
        return weekly_data, monthly_data

# Create functions for app interface
def main():
    st.title("Electricity Top-Up App")

    # Login
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in user_database and password == user_database[username]["password"]:
            st.success("Logged in successfully!")
            session_state.logged_in = True
            session_state.username = username
            session_state.is_admin = user_database[username]["is_admin"]
        else:
            st.error("Invalid username or password!")

    # Registration (for demo purposes, not recommended for production)
    st.subheader("Registration")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if new_username and new_password:
            if new_username not in user_database:
                user_database[new_username] = {"password": new_password, "is_admin": False, "bank_name": "", "account_number": "", "balance": 0.0, "remaining_units": 0, "threshold": 0}
                st.success("Registered successfully!")
            else:
                st.error("Username already exists!")
        else:
            st.warning("Please enter a username and password!")

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if st.session_state.logged_in:
        # Show app only if logged in
        st.subheader(f"Welcome, {st.session_state.username}!")

        if st.session_state.is_admin:
            # Admin view
            st.header("Admin Overview")

            # Display consumption overview for all users (excluding banking details)
            for username, user_data in user_database.items():
                if username != "admin":
                    st.subheader(f"User: {username}")
                    st.write(f"Remaining Units: {user_data['remaining_units']}")
                    st.write(f"Threshold: {user_data['threshold']}")
                    st.write("---")
        else:
            # User view
            st.header("Electricity Top-Up")

            # Get user data
            app = ElectricityTopUpApp(st.session_state.username)

            # Operations: Top Up and Consume Units
            operation = st.selectbox("Select Operation", ["Top Up", "Consume Units"])
            units = st.number_input("Enter Units", min_value=0, step=1)
            if operation == "Top Up":
                meter_number = st.text_input("Enter Meter Number", "1234567890")
                if st.button("Top Up"):
                    app.top_up(units, meter_number)
            elif operation == "Consume Units":
                if st.button("Consume Units"):
                    app.consume_units(units)

            # Display Bank Details (for demonstration)
            st.subheader("Current Bank Details")
            st.write(f"Bank Name: {app.bank_name}")
            st.write(f"Account Number: {app.account_number}")
            st.write(f"Balance: {app.balance}")

            # Create a new page for checking remaining units and visualization
            st.sidebar.title("User Dashboard")
            page = st.sidebar.radio("Select Page", ["Remaining Units", "Consumption Visualization"])

            if page == "Remaining Units":
                st.header("Remaining Units")
                st.write(f"Remaining Units: {app.remaining_units}")

            elif page == "Consumption Visualization":
                st.header("Consumption Visualization")

                # Get consumption data
                weekly_data, monthly_data = app.get_consumption_data()

                # Plot weekly consumption
                st.subheader("Weekly Consumption")
                weekly_fig = go.Figure()
                weekly_fig.add_trace(go.Bar(x=weekly_data["Date"], y=weekly_data["Units"], name="Units", marker_color="blue"))
                weekly_fig.add_trace(go.Scatter(x=weekly_data["Date"], y=weekly_data["Cost"], mode="lines", name="Cost (R)", yaxis="y2", line=dict(color="red")))
                weekly_fig.update_layout(title="Weekly Consumption", xaxis_title="Date", yaxis_title="Units", yaxis2=dict(title="Cost (R)", overlaying="y", side="right"), legend=dict(x=0, y=1.0))
                st.plotly_chart(weekly_fig)

                # Plot monthly consumption
                st.subheader("Monthly Consumption")
                monthly_fig = go.Figure()
                monthly_fig.add_trace(go.Bar(x=monthly_data["Date"], y=monthly_data["Units"], name="Units", marker_color="green"))
                monthly_fig.add_trace(go.Scatter(x=monthly_data["Date"], y=monthly_data["Cost"], mode="lines", name="Cost (R)", yaxis="y2", line=dict(color="orange")))
                monthly_fig.update_layout(title="Monthly Consumption", xaxis_title="Date", yaxis_title="Units", yaxis2=dict(title="Cost (R)", overlaying="y", side="right"), legend=dict(x=0, y=1.0))
                st.plotly_chart(monthly_fig)

if __name__ == "__main__":
    main()
