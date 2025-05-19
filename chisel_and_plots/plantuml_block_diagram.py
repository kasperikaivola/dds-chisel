import zlib, base64
plantuml_code = """
@startuml
left to right direction

' General styling
skinparam defaultTextAlignment center
skinparam shadowing false
skinparam linetype ortho
left to right direction
skinparam rectangle {   
    FontSize 12
    FontColor Black
    BorderColor Black
}

' Stereotype styles
skinparam component {
    BackgroundColor<<input>> #D6EAF8
    BorderColor<<input>> #2980B9

    BackgroundColor<<output>> #D5F5E3
    BorderColor<<output>> #27AE60

    BackgroundColor<<logic>> #F9E79F
    BorderColor<<logic>> #B7950B
}

' Inputs and outputs
rectangle "<<input>> SYSTEM CLOCK" as CLK
rectangle "<<output>> D/A CONVERTER\n(12 bits)" as DAC

' DDS block
rectangle "DDS Implementation" as CORE {
    top to bottom direction
    rectangle "<<input>> TUNING WORD M\n(32 bits)" as TWM
    rectangle "<<logic>> PHASE ACCUMULATOR\n(32 bits)" as PA
    rectangle "<<logic>> PHASE REGISTER" as PR
    rectangle "<<logic>> PHASE-TO-AMPLITUDE CONVERTER\n(10-bit LUT)" as PAC
}
rectangle "<<output>> f_out" as FOUT
' Connections
TWM --> PA
PA --> PAC
PR --> PA
PAC --> DAC
DAC --> FOUT

CLK --> PA
CLK --> PR
CLK --> DAC

@enduml
"""

# Encode the updated PlantUML code
compressed_updated = zlib.compress(plantuml_code.encode('utf-8'), 9)
encoded_updated = base64.urlsafe_b64encode(compressed_updated).decode('ascii')

# Generate the final URL
kroki_url_updated = f"https://kroki.io/plantuml/svg/{encoded_updated}"
kroki_url_updated
