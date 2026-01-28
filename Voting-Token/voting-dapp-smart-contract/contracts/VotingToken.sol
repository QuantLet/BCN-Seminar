// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract VotingToken is ERC20, Ownable {
    constructor() ERC20("QNARReward", "QNAR") Ownable(msg.sender) {
        // Mint initial supply to the deployer
        _mint(msg.sender, 1000000 * 10 ** decimals());
    }

    // Req 10: Admin can add more tokens (Minting)
    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }
}
