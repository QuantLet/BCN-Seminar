// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract VotingSystem is Ownable {
    IERC20 public rewardToken;

    // --- STRUCTS ---

    struct Candidate {
        uint256 id;
        string name;
        uint256 voteCount;
        address payable wallet;
        address[] voters;
    }

    struct Round {
        uint256 id;
        string title;
        uint256 endTime;
        bool isActive;
        uint256 totalRewardPool;
    }

    struct RewardRecord {
        uint256 roundId;
        string roundTitle;
        uint256 amount;
        uint256 rank; // 1, 2, 3 for winners. 0 for Voter Participation.
        uint256 timestamp;
    }

    // --- STATE VARIABLES ---

    uint256 public roundCount;

    mapping(uint256 => Round) public rounds;
    mapping(uint256 => Candidate[]) public roundCandidates;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    // Prevent duplicate candidates in a specific round
    mapping(uint256 => mapping(address => bool)) public isCandidateInRound;

    // Track all voters for a specific round ID to distribute rewards later
    mapping(uint256 => address[]) public roundVoters;

    mapping(address => RewardRecord[]) public userRewards;
    mapping(address => string) public usernames;

    // --- EVENTS ---
    event RoundCreated(uint256 roundId, string title, uint256 rewardPool);
    event Voted(uint256 roundId, address voter, uint256 candidateId);
    event RoundEnded(uint256 roundId, address[3] winners);
    event RewardDistributed(
        address indexed user,
        uint256 amount,
        uint256 roundId,
        string rewardType
    );

    constructor(address _tokenAddress) Ownable(msg.sender) {
        rewardToken = IERC20(_tokenAddress);
    }

    // --- USER MANAGEMENT ---
    function setUsername(string memory _name) external {
        usernames[msg.sender] = _name;
    }

    // --- ADMIN FUNCTIONS ---

    function createRound(
        string memory _title,
        uint256 _durationInSeconds,
        uint256 _rewardPool
    ) external onlyOwner {
        roundCount++;
        rounds[roundCount] = Round(
            roundCount,
            _title,
            block.timestamp + _durationInSeconds,
            true,
            _rewardPool
        );
        emit RoundCreated(roundCount, _title, _rewardPool);
    }

    function addCandidate(
        uint256 _roundId,
        string memory _name,
        address payable _wallet
    ) external onlyOwner {
        require(rounds[_roundId].isActive, "Round not active");

        // CHECK: Prevent adding the same person twice
        require(
            !isCandidateInRound[_roundId][_wallet],
            "Candidate already exists in this round"
        );

        uint256 id = roundCandidates[_roundId].length;
        address[] memory emptyVoters;

        roundCandidates[_roundId].push(
            Candidate(id, _name, 0, _wallet, emptyVoters)
        );

        // Mark them as added
        isCandidateInRound[_roundId][_wallet] = true;
    }

    // --- REWARD DISTRIBUTION LOGIC ---
    function endRound(uint256 _roundId) external onlyOwner {
        Round storage round = rounds[_roundId];
        require(round.isActive, "Round already ended");
        round.isActive = false;

        uint256 contractBalance = rewardToken.balanceOf(address(this));
        require(
            contractBalance >= round.totalRewardPool,
            "Insufficient contract balance"
        );

        Candidate[] storage candidates = roundCandidates[_roundId];

        // 1. Sort Candidates (Bubble Sort - Highest Votes First)
        for (uint x = 0; x < candidates.length; x++) {
            for (uint y = x + 1; y < candidates.length; y++) {
                if (candidates[y].voteCount > candidates[x].voteCount) {
                    Candidate memory temp = candidates[x];
                    candidates[x] = candidates[y];
                    candidates[y] = temp;
                }
            }
        }

        // --- PART A: Distribute 90% Pool to Top 3 Ranks ---
        uint256[3] memory rankPercentages = [
            uint256(40),
            uint256(30),
            uint256(20)
        ];
        address[3] memory winners;

        uint256 currentRankSlot = 0;
        uint256 i = 0;

        while (i < candidates.length && currentRankSlot < 3) {
            if (candidates[i].voteCount == 0) {
                i++;
                continue;
            }

            // Identify Tie Group
            uint256 groupSize = 1;
            for (uint256 j = i + 1; j < candidates.length; j++) {
                if (candidates[j].voteCount == candidates[i].voteCount) {
                    groupSize++;
                } else {
                    break;
                }
            }

            // Sum percentages
            uint256 totalPercentForGroup = 0;
            for (uint256 k = 0; k < groupSize; k++) {
                if (currentRankSlot + k < 3) {
                    totalPercentForGroup += rankPercentages[
                        currentRankSlot + k
                    ];
                }
            }

            // Distribute
            if (totalPercentForGroup > 0) {
                uint256 totalGroupReward = (round.totalRewardPool *
                    totalPercentForGroup) / 100;
                uint256 sharePerPerson = totalGroupReward / groupSize;

                for (uint256 k = 0; k < groupSize; k++) {
                    uint256 candidateIndex = i + k;
                    Candidate memory winner = candidates[candidateIndex];

                    rewardToken.transfer(winner.wallet, sharePerPerson);

                    if (candidateIndex < 3)
                        winners[candidateIndex] = winner.wallet;

                    userRewards[winner.wallet].push(
                        RewardRecord({
                            roundId: _roundId,
                            roundTitle: round.title,
                            amount: sharePerPerson,
                            rank: currentRankSlot + 1,
                            timestamp: block.timestamp
                        })
                    );

                    emit RewardDistributed(
                        winner.wallet,
                        sharePerPerson,
                        _roundId,
                        "CANDIDATE_WIN"
                    );
                }
            }

            currentRankSlot += groupSize;
            i += groupSize;
        }

        // --- PART B: Distribute 10% to ALL Voters ---
        address[] memory allVoters = roundVoters[_roundId];

        if (allVoters.length > 0) {
            uint256 voterPool = (round.totalRewardPool * 10) / 100;
            uint256 amountPerVoter = voterPool / allVoters.length;

            if (amountPerVoter > 0) {
                for (uint j = 0; j < allVoters.length; j++) {
                    rewardToken.transfer(allVoters[j], amountPerVoter);

                    userRewards[allVoters[j]].push(
                        RewardRecord({
                            roundId: _roundId,
                            roundTitle: round.title,
                            amount: amountPerVoter,
                            rank: 0,
                            timestamp: block.timestamp
                        })
                    );

                    emit RewardDistributed(
                        allVoters[j],
                        amountPerVoter,
                        _roundId,
                        "VOTER_REWARD"
                    );
                }
            }
        }

        emit RoundEnded(_roundId, winners);
    }

    function fundContract(uint256 amount) external {
        rewardToken.transferFrom(msg.sender, address(this), amount);
    }

    // --- VOTING FUNCTIONS ---

    function vote(uint256 _roundId, uint256 _candidateId) external {
        // 1. Check Admin Status
        require(msg.sender != owner(), "Admin cannot vote");

        // --- CHANGE: I REMOVED THE LINE THAT BLOCKED CANDIDATES FROM VOTING ---
        // Candidates are now allowed to pass this point.

        // 2. Standard Checks
        require(rounds[_roundId].isActive, "Round ended");
        require(block.timestamp < rounds[_roundId].endTime, "Time expired");
        require(!hasVoted[_roundId][msg.sender], "Already voted");

        Candidate storage candidate = roundCandidates[_roundId][_candidateId];

        // 3. Prevent Self-Voting
        // This line ensures a candidate cannot select themselves
        require(candidate.wallet != msg.sender, "Cannot vote for yourself");

        candidate.voteCount++;
        candidate.voters.push(msg.sender);
        hasVoted[_roundId][msg.sender] = true;
        roundVoters[_roundId].push(msg.sender);

        emit Voted(_roundId, msg.sender, _candidateId);
    }

    // --- VIEW FUNCTIONS ---

    function getCandidates(
        uint256 _roundId
    ) external view returns (Candidate[] memory) {
        return roundCandidates[_roundId];
    }

    function getCandidateVoters(
        uint256 _roundId,
        uint256 _candidateId
    ) external view returns (address[] memory) {
        return roundCandidates[_roundId][_candidateId].voters;
    }

    function getUserRewardHistory(
        address _user
    ) external view returns (RewardRecord[] memory) {
        return userRewards[_user];
    }

    function isAdmin(address _user) external view returns (bool) {
        return _user == owner();
    }
}
