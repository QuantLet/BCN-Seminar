# Voting App Frontend

## Purpose
The Voting App is a decentralized application (dApp) designed to facilitate transparent and secure voting processes. Built on blockchain technology, it ensures that votes are immutable, verifiable, and tamper-proof. This application is ideal for governance, community decision-making, and other scenarios requiring trustless voting mechanisms.

## Features
- **Explore:** Browse through various voting proposals.
- **Governance:** Participate in governance by creating and voting on proposals.
- **Leaderboard:** View top contributors and their achievements.
- **Rewards:** Check rewards earned through participation.

## How It Works
1. **Connect Wallet:** Users connect their Web3 wallet (e.g., MetaMask) to interact with the application.
2. **Explore Proposals:** Navigate to the "Explore" section to view ongoing and past proposals.
3. **Vote:** Participate in voting by selecting a proposal and casting your vote.
4. **Create Proposals:** Users with the required permissions can create new proposals in the "Governance" section.
5. **Leaderboard and Rewards:** Check the leaderboard for top contributors and claim rewards in the rewards section.

## Instructions

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- A Web3 wallet (e.g., MetaMask)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd voting-app-frontend/frontend
   ```
3. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

### Running the Application
1. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

### Building for Production
1. Build the application:
   ```bash
   npm run build
   # or
   yarn build
   ```
2. Start the production server:
   ```bash
   npm start
   # or
   yarn start
   ```

### Folder Structure
- **app/**: Contains the main application pages and subdirectories for different sections (e.g., explore, governance, rewards).
- **components/**: Reusable UI components like Navbar and Footer.
- **context/**: Context providers for managing global state (e.g., UIContext, Web3Context).
- **public/**: Static assets like images.

### Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.