# Face-Based Access Control System
The purpose of this project is to make a usable system that 
- Allows software users to upload a user profile consististing of biometric data 
- Consumes streaming data (frames) from a streaming camera 
- If an approved user is identified with confience above a certain level, the state of the lock is changed to open

## Design 
Given the lack of GPU resources and for scope, this project will consist only of deploying a SOTA model
and operationalizing the system so that successful operation commences

### Package management 
Many ML/DS projects leverage conda for preconfigured stacks of libraries and a familiar api for management. 
This project will make use of uv for super speedy package management and a more lightweight surface for containerization. 
