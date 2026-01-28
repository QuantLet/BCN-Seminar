# Voting DApp

## Purpose
The Voting DApp is a decentralized application designed to facilitate secure, transparent, and tamper-proof voting processes using blockchain technology. By leveraging smart contracts, this application ensures that votes are immutable and verifiable, eliminating the need for trust in a central authority.

## How It Works
1. **Smart Contracts**: The application uses two main smart contracts:
   - `VotingSystem.sol`: Manages the voting process, including creating proposals, casting votes, and tallying results.
   - `VotingToken.sol`: Implements a token system to represent voting rights.

2. **Deployment**: The smart contracts are deployed to a blockchain network, ensuring transparency and immutability.

3. **Voting Process**:
   - Users are assigned voting tokens.
   - Proposals are created, and users can cast their votes using their tokens.
   - Votes are recorded on the blockchain, and results are calculated automatically.

4. **Frontend Interaction**: Users interact with the DApp through a user-friendly interface that communicates with the blockchain via a wallet (e.g., MetaMask).

## Instructions

### Prerequisites
- Node.js installed on your system.
- A blockchain wallet (e.g., MetaMask) configured.
- Access to a blockchain network (e.g., Ethereum testnet).

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd voting-dapp
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Compilation
Compile the smart contracts using Hardhat:
```bash
npx hardhat compile
```

### Deployment
Deploy the smart contracts to a blockchain network:
```bash
npx hardhat run scripts/deploy.js --network <network-name>
```
Replace `<network-name>` with your desired network (e.g., `sepolia`).

### Testing
Run the tests to ensure the contracts work as expected:
```bash
npx hardhat test
```

### Interaction
1. Start the DApp:
   ```bash
   npm start
   ```
2. Open your browser and navigate to the provided URL.
3. Connect your wallet and interact with the application.

### Folder Structure
- `contracts/`: Contains the smart contracts.
- `scripts/`: Deployment and interaction scripts.
- `test/`: Test scripts for the smart contracts.
- `artifacts/` and `cache/`: Generated files during compilation and deployment.

### Notes
- Ensure your wallet is connected to the same network where the contracts are deployed.
- Use the `ignition/deployments/` folder to find deployed contract addresses.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.