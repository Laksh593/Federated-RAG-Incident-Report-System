import os
from pathlib import Path

# Expanded dataset of policies to make the knowledge base robust
policies = {
    "protocol_active_shooter.txt": """
SECURITY PROTOCOL: ACTIVE SHOOTER / ARMED INTRUDER (CODE RED)

1. INITIAL RESPONSE
   - Immediately activate the facility lockdown alarm.
   - Dial 911 or the local emergency number to report the incident. Provide location, number of shooters, physical description, number and types of weapons, and number of potential victims.
   - Notify the facility security team via the emergency radio channel.

2. RUN, HIDE, FIGHT
   - RUN: Have an escape route and plan in mind. Leave your belongings behind. Keep your hands visible.
   - HIDE: Hide in an area out of the shooter's view. Block entry to your hiding place and lock the doors. Silence your cell phone and/or pager.
   - FIGHT: As a last resort and only when your life is in imminent danger, attempt to incapacitate the shooter. Act with physical aggression and throw items at the active shooter.

3. EVACUATION AND RECOVERY
   - Follow instructions from law enforcement once they arrive.
   - Keep hands visible and empty.
   - Proceed to the designated safe assembly area for headcounts.
   """,

    "protocol_drone_incursion.txt": """
SECURITY PROTOCOL: UNAUTHORIZED DRONE / UAV INCURSION (CODE SKIES)

1. DETECTION
   - Ensure the drone is actually within restricted airspace (typically within 500 feet of the facility perimeter).
   - Document the drone's physical description (size, color, number of rotors, any visible payload or camera).
   - Note the direction from which the drone originated and the direction it travels.

2. RESPONSE ACTIONS
   - Do NOT attempt to shoot down, physically intercept, or jam the drone's signal unless authorized by federal aviation authorities (e.g., FAA).
   - Notify the shift supervisor and the local aviation authority immediately.
   - If the drone appears to be conducting surveillance, lockdown sensitive outdoor areas (e.g., loading docks, outdoor testing facilities).
   - Dispatch a security sweep team to search the immediate perimeter (up to 2 miles) for the drone operator.

3. REPORTING
   - File an incident report documenting the exact time, duration, and behavior of the drone. Provide any video evidence captured by ground cameras.
   """,

    "protocol_medical_emergency.txt": """
SECURITY PROTOCOL: MEDICAL EMERGENCY / CARDIAC EVENT (CODE BLUE)

1. IMMEDIATE ASSESSMENT
   - Verify the nature of the medical emergency. Check for responsiveness, breathing, and pulse if trained to do so.
   - Do not move the victim unless they are in immediate, life-threatening danger (e.g., fire, collapsing structure).

2. ACTIVATION
   - Call emergency medical services (EMS) immediately (911).
   - Notify the on-site medical response team or designated CPR/AED trained personnel via radio.
   - Retrieve the nearest Automated External Defibrillator (AED) and First Aid kit.

3. INTERVENTION
   - If the victim is unresponsive and not breathing normally, begin CPR immediately (hard and fast in the center of the chest).
   - Turn on the AED and follow the voice prompts.
   - Clear the area of unnecessary personnel to afford the victim privacy and to allow EMS unimpeded access.

4. POST-INCIDENT
   - Assign a staff member to meet EMS at the main entrance to guide them quickly to the victim.
   - Secure any personal belongings of the victim and document the actions taken before EMS arrival.
   """,

    "protocol_chemical_spill.txt": """
SECURITY PROTOCOL: HAZARDOUS MATERIAL / CHEMICAL SPILL (CODE ORANGE)

1. ISOLATION
   - Immediately evacuate all personnel from the immediate vicinity of the spill.
   - Do not attempt to clean up the spill unless you are specifically trained and equipped with the correct Personal Protective Equipment (PPE) for the identified chemical.
   - Close doors to the affected area to isolate fumes or vapors.

2. IDENTIFICATION
   - Attempt to identify the spilled substance from a safe distance (e.g., reading labels, manifesting). Ensure the Safety Data Sheet (SDS) is retrieved.
   - Notify the facility Hazardous Materials (HAZMAT) response team and the environmental health and safety (EHS) officer.

3. CONTAINMENT AND NOTIFICATION
   - If trained and safe to do so, shut off HVAC systems in the affected zone to prevent the spread of toxic fumes.
   - If the spill is highly toxic, flammable, or exceeds 5 gallons, immediately pull the fire alarm to trigger a building-wide evacuation and call 911.

4. CLEANUP
   - Only authorized HAZMAT contractors or properly equipped internal teams may commence cleanup operations following the specific SDS guidelines for the chemical.
   """,
   
    "protocol_trespassing.txt": """
SECURITY PROTOCOL: TRESPASSING / UNAUTHORIZED ACCESS (CODE YELLOW)

1. VERIFICATION
   - Confirm via camera or visual inspection that the individual lacks proper identification badges or authorized escort.
   - Note the individual's physical description, clothing, and any carried items (e.g., bags, tools).

2. INTERCEPT AND ESCORT
   - Approach the individual calmly but firmly. Ask for identification and their purpose for being in the restricted area.
   - If they are simply lost, politely escort them back to the public reception area or their designated authorized zone.
   - If they refuse to comply, become belligerent, or attempt to flee deeper into the facility, immediately call for backup security teams.

3. ESCALATION
   - Do NOT engage in physical altercations unless the individual poses an immediate physical threat to staff or critical infrastructure.
   - If the trespasser becomes aggressive or refuses to leave after repeated warnings, contact local law enforcement to perform a formal trespass eviction.

4. DOCUMENTATION
   - Log the individual's description and actions. Review access logs and camera footage to determine how the unauthorized access occurred (e.g., tailgating, broken lock) and submit a vulnerability report.
   """
}

def generate_policies(kb_dir: str):
    path = Path(kb_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    for filename, content in policies.items():
        file_path = path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Generated {filename}")

if __name__ == "__main__":
    # Ensure this runs relative to the project root
    project_root = Path(__file__).parent.parent
    kb_dir = project_root / "data" / "knowledge_base"
    print(f"Generating synthetic policies in {kb_dir}...")
    generate_policies(kb_dir)
    print("Done!")
