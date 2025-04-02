import sqlite3
import os

DB_PATH = "fracture_treatment.db"

def create_database():
    # Remove existing database file to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create treatment recommendations table
    cursor.execute('''CREATE TABLE IF NOT EXISTS Treatment_Recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fracture_pattern TEXT,
                        displacement_severity TEXT,
                        joint_involvement TEXT,
                        treatment TEXT
                    )''')

    # Comprehensive treatment recommendations for all combinations
    treatments = [
        # Simple fractures
        ("Simple", "Minimal", "No", "Conservative treatment with cast or splint immobilization for 4-6 weeks. Regular follow-up x-rays."),
        ("Simple", "Minimal", "Yes", "Surgical fixation required to restore joint congruity. Early physical therapy after healing."),
        ("Simple", "Moderate", "No", "Closed reduction followed by casting or functional bracing. 6-8 weeks immobilization."),
        ("Simple", "Moderate", "Yes", "ORIF (Open Reduction Internal Fixation) with screws or plates to restore joint alignment."),
        ("Simple", "Severe", "No", "ORIF (Open Reduction Internal Fixation) with plates or screws. Post-op rehabilitation."),
        ("Simple", "Severe", "Yes", "ORIF with anatomic reduction. Consider arthroscopic assessment and possible joint reconstruction."),
        
        # Wedge fractures
        ("Wedge", "Minimal", "No", "Closed reduction and casting. Consider surgical fixation if instability present."),
        ("Wedge", "Minimal", "Yes", "ORIF to stabilize the wedge fragment and restore joint surface."),
        ("Wedge", "Moderate", "No", "ORIF with buttress plating recommended to prevent fragment displacement."),
        ("Wedge", "Moderate", "Yes", "ORIF with anatomic reduction of articular surface. Early ROM exercises after stability achieved."),
        ("Wedge", "Severe", "No", "ORIF with plates and screws. Consider bone grafting if bone loss present."),
        ("Wedge", "Severe", "Yes", "Complex reconstruction with ORIF, possibly staged procedures if severe comminution."),
        
        # Comminuted fractures
        ("Comminuted", "Minimal", "No", "ORIF or external fixation depending on soft tissue condition. Limited weight bearing."),
        ("Comminuted", "Minimal", "Yes", "ORIF with special attention to articular surface reconstruction. Consider external fixation."),
        ("Comminuted", "Moderate", "No", "ORIF with bridging plate or external fixation. Consider bone grafting."),
        ("Comminuted", "Moderate", "Yes", "ORIF with anatomic reconstruction of joint surface. May require specialized implants."),
        ("Comminuted", "Severe", "No", "External fixation or ORIF with locked plates. Staged procedures may be necessary."),
        ("Comminuted", "Severe", "Yes", "Complex surgical reconstruction. Consider spanning external fixation initially, followed by definitive ORIF."),
        
        # Fallback options for any unspecified combinations
        ("Any", "Any", "Yes", "Surgical intervention recommended due to joint involvement. Consult orthopedic specialist immediately."),
        ("Any", "Severe", "Any", "Urgent surgical consultation required. Temporary stabilization followed by definitive fixation."),
        ("Any", "Any", "Any", "Individualized treatment based on fracture assessment. Consult orthopedic specialist for specific recommendations.")
    ]

    cursor.executemany('''INSERT INTO Treatment_Recommendations 
                          (fracture_pattern, displacement_severity, joint_involvement, treatment) 
                          VALUES (?, ?, ?, ?)''', treatments)

    conn.commit()
    conn.close()
    print("Database created and populated successfully with comprehensive treatment options.")

def get_treatment_recommendation(fracture_pattern, displacement_severity, joint_involvement):
    """
    Get treatment recommendation for a specific fracture configuration.
    
    Args:
        fracture_pattern (str): 'Simple', 'Wedge', or 'Comminuted'
        displacement_severity (str): 'Minimal', 'Moderate', or 'Severe'
        joint_involvement (str): 'Yes' or 'No'
        
    Returns:
        str: Treatment recommendation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Debug print to verify input values
    print(f"Looking up recommendation for: Pattern={fracture_pattern}, Displacement={displacement_severity}, Joint={joint_involvement}")

    # Try exact match first
    cursor.execute('''SELECT treatment FROM Treatment_Recommendations 
                    WHERE fracture_pattern = ? 
                    AND displacement_severity = ? 
                    AND joint_involvement = ?''', 
                    (fracture_pattern, displacement_severity, joint_involvement))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # Try with joint involvement wildcard
    cursor.execute('''SELECT treatment FROM Treatment_Recommendations 
                    WHERE fracture_pattern = ? 
                    AND displacement_severity = ? 
                    AND joint_involvement = 'Any' ''', 
                    (fracture_pattern, displacement_severity))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # Try with displacement wildcard
    cursor.execute('''SELECT treatment FROM Treatment_Recommendations 
                    WHERE fracture_pattern = ? 
                    AND displacement_severity = 'Any' 
                    AND (joint_involvement = ? OR joint_involvement = 'Any')''', 
                    (fracture_pattern, joint_involvement))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # Try with pattern wildcard
    cursor.execute('''SELECT treatment FROM Treatment_Recommendations 
                    WHERE fracture_pattern = 'Any' 
                    AND (displacement_severity = ? OR displacement_severity = 'Any')
                    AND (joint_involvement = ? OR joint_involvement = 'Any')''', 
                    (displacement_severity, joint_involvement))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # Fallback to most generic recommendation
    cursor.execute('''SELECT treatment FROM Treatment_Recommendations 
                    WHERE fracture_pattern = 'Any' 
                    AND displacement_severity = 'Any' 
                    AND joint_involvement = 'Any' ''')
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else "Consult with orthopedic specialist for personalized treatment plan."

def test_database():
    """
    Test function to verify all combinations return valid recommendations
    """
    patterns = ['Simple', 'Wedge', 'Comminuted']
    displacements = ['Minimal', 'Moderate', 'Severe']
    joint_involvements = ['Yes', 'No']
    
    print("Testing all treatment recommendation combinations:")
    for pattern in patterns:
        for displacement in displacements:
            for joint in joint_involvements:
                recommendation = get_treatment_recommendation(pattern, displacement, joint)
                print(f"Pattern: {pattern}, Displacement: {displacement}, Joint: {joint}")
                print(f"Recommendation: {recommendation}")
                print("-" * 50)

# Run to create/recreate the database and test all combinations
if __name__ == "__main__":
    create_database()
    test_database()