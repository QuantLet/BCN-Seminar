import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

// Define 1000 Tokens (assuming 18 decimals) using BigInt
const FUNDING_AMOUNT = 1000n * 10n ** 18n;

export default buildModule("VotingModule", (m) => {
  // 1. Deploy the ERC-20 Reward Token
  // This returns a future representing the deployed contract instance
  const token = m.contract("VotingToken");

  // 2. Deploy the Voting System
  // We pass the 'token' future directly. Ignition resolves the address automatically.
  const voting = m.contract("VotingSystem", [token]);

  // 3. Approve the Voting Contract to spend the Admin's tokens
  // We call the 'approve' function on the 'token' contract.
  const approveTx = m.call(token, "approve", [voting, FUNDING_AMOUNT]);

  // 4. Fund the Voting Contract
  // We call 'fundContract' on the 'voting' contract.
  // CRITICAL: We add { after: [approveTx] } to ensure approval happens first.
  const fundTx = m.call(voting, "fundContract", [FUNDING_AMOUNT], {
    after: [approveTx],
  });

  // Return the contracts so we can use them in tests or see addresses
  return { token, voting };
});