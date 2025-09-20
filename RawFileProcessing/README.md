# Install MongoDB Community Edition

This document provides step-by-step instructions to install MongoDB on various platforms.

## Table of Contents
- [Install MongoDB Community Edition](#install-mongodb-community-edition)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installing MongoDB](#installing-mongodb)
    - [Windows Installation](#windows-installation)
    - [macOS Installation](#macos-installation)
    - [Linux Installation](#linux-installation)
      - [For Ubuntu](#for-ubuntu)
  - [Pushing data to MongoDB](#pushing-data-to-mongodb)
    - [Using MongoDB to Store SDR data](#using-mongodb-to-store-sdr-data)
      - [Steps to Get Started:](#steps-to-get-started)

---

## Prerequisites

Before installing MongoDB, make sure that:
- You have administrative or root access to the system.
- You have an active internet connection to download MongoDB packages.

## Installing MongoDB

### Windows Installation
I don't support Windows.

### macOS Installation

Follow the instruction [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/#install-mongodb-community-edition-on-macos)

### Linux Installation

For the most updated istruction please follow [this link](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/#install-mongodb-community-edition-on-ubuntu). 

#### For Ubuntu


1. **Import the public key**
   ```bash
   sudo apt-get install gnupg curl
   curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
   --dearmor
2. **Create a list of file for MongoDB**
    Create the `/etc/apt/sources.list.d/mongodb-org-7.0.list` file for Ubuntu 22.04
   ```bash
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
   ```
3. **Reload local package database**
    ```bash
    sudo apt-get update
    ```
4. **Install MongoDB**
   ```bash
   sudo apt-get install -y mongodb-org
   ```
5. **Start the MongoDB**
   ```bash 
   sudo systemctl start mongod
   ```
6. **Connection String**
   `mongodb://localhost:27017/`

## Pushing data to MongoDB
Here's an improved version of the text for the README:

---

### Using MongoDB to Store SDR data

After successfully installing MongoDB on your system, you can now use the [`pushToMongoDB.ipynb`](pushToMongoDB.ipynb) Jupyter notebook to create collections and store SDR recordings in your MongoDB database.

#### Steps to Get Started:

1. **Prepare the Dataset Folder**:
   - Create a folder called `DataSet` in the root directory of this project.
   - Download the **SDR1** and **SDR2** datasets from IEEE Dataport and place them in the `DataSet` folder.

2. **Running the Notebook**:
   - Open the [`pushToMongoDB.ipynb`](pushToMongoDB.ipynb) Jupyter notebook.
   - Follow the steps in the notebook to connect to your MongoDB instance and insert metadata into the relevant collections.





   
