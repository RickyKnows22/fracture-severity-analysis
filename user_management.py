import sqlite3
import hashlib

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def view_users():
    """Retrieve and print all user accounts from the database."""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT username, email FROM users")
        users = cursor.fetchall()
        
        print("\nRegistered Users:")
        print("----------------")
        if not users:
            print("No users found in the database.")
        else:
            for idx, (username, email) in enumerate(users, 1):
                print(f"{idx}. Username: {username}, Email: {email}")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        conn.close()

def delete_user():
    """Delete a user from the database."""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        # First, view users
        cursor.execute("SELECT username, email FROM users")
        users = cursor.fetchall()
        
        if not users:
            print("No users to delete.")
            return
        
        # Print users with indices
        print("\nUsers:")
        for idx, (username, email) in enumerate(users, 1):
            print(f"{idx}. Username: {username}, Email: {email}")
        
        # Select user to delete
        choice = int(input("\nEnter the number of the user to delete: ")) - 1
        
        if 0 <= choice < len(users):
            username_to_delete = users[choice][0]
            
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete user {username_to_delete}? (yes/no): ")
            
            if confirm.lower() in ['yes', 'y']:
                cursor.execute("DELETE FROM users WHERE username = ?", (username_to_delete,))
                conn.commit()
                print(f"User {username_to_delete} has been deleted.")
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid selection.")
    
    except ValueError:
        print("Please enter a valid number.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        conn.close()

def add_user():
    """Add a new user to the database."""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    try:
        username = input("Enter username: ")
        email = input("Enter email: ")
        password = input("Enter password: ")
        
        # Hash the password
        hashed_password = hash_password(password)
        
        # Insert the new user
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                       (username, email, hashed_password))
        conn.commit()
        print("User added successfully!")
    
    except sqlite3.IntegrityError:
        print("Username or email already exists.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        conn.close()

def main_menu():
    """Display main menu for user management."""
    while True:
        print("\n--- User Management ---")
        print("1. View Users")
        print("2. Add User")
        print("3. Delete User")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            view_users()
        elif choice == '2':
            add_user()
        elif choice == '3':
            delete_user()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Ensure users table exists
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, 
                      email TEXT UNIQUE, 
                      password TEXT)''')
    conn.close()
    
    # Start the menu
    main_menu()