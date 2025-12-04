function env = createAttitudeControlEnv()
% Creates the Attitude Control Environment using rlFunctionEnv
% Defines parameters, specifications, and nested step/reset functions.

    % --- Define Environment Specifications ---
    stateDim = 7;
    actionDim = 3;
    maxTorqueLimit = 0.5; % Define the ACTUAL max torque for the environment action spec

    obsInfo = rlNumericSpec([stateDim 1]);
    obsInfo.Name = 'observations';
    obsInfo.Description = '[q1; q2; q3; q4; wx; wy; wz]'; % Quaternion error then omega

    % Action space defined by the physical torque limits
    actInfo = rlNumericSpec([actionDim 1], 'LowerLimit', -maxTorqueLimit, 'UpperLimit', maxTorqueLimit);
    actInfo.Name = 'torque';
    actInfo.Description = '[Mx; My; Mz]';

    % --- Define Parameters Struct ---
    % This struct is accessible by the nested functions defined below
    params.I = [0.872; 0.115; 0.797]; % Principal moments of inertia (COLUMN vector)
    params.h = 1/240;                 % Simulation time step (s)
    params.torqueScale = 0.5;         % Scale factor if agent outputs [-1, 1] actions
    params.frameskipNum = 19;         % Additional free rotation steps (total steps = 1 + frameskipNum)
    params.maxAngleSlew = 150;        % Max random initial angle (degrees)
    params.minAngleSlew = 30;         % Min random initial angle (degrees)
    params.alpha = [-1; -1; -1; 1];   % Reward weights for quaternion components squared
    params.maxEpisodeSteps = 500;     % Max agent steps before truncation
    params.maxAngVelo = 0.5;          % Max angular velocity limit (rad/s) for termination
    params.rewardThresholdQs = cos(deg2rad(0.25)/2); % Scalar quaternion threshold for reward switch


    % --- Create Environment using Function Handles to NESTED functions ---
    % Only pass the four required arguments to rlFunctionEnv
    env = rlFunctionEnv(obsInfo, actInfo, @nestedStepFunction, @nestedResetFunction);


    % =========================================================================
    % NESTED CALLBACK FUNCTIONS (Defined inside the main function)
    % They automatically have access to obsInfo, actInfo, params defined above
    % =========================================================================

    function [NextObservation, Reward, IsDone, LoggedSignals] = nestedStepFunction(Action, LoggedSignals)
        % Implements the step logic for the environment

        % --- Extract State from LoggedSignals ---
        quat = LoggedSignals.quat;                  % Current error quaternion [q1;q2;q3;q4]
        omega = LoggedSignals.omega;                % Current angular velocity [wx;wy;wz]
        nSteps = LoggedSignals.nSteps;              % Current agent step count
        qsPrev = LoggedSignals.qsPrev;              % Previous scalar quaternion part (q4)
        stepsBeyondDone = LoggedSignals.stepsBeyondDone; % Counter for steps after termination/truncation
        isInStabilizationPhase = LoggedSignals.isInStabilizationPhase; % Flag for reward phase
        LoggedSignals.MaxSteps = params.maxEpisodeSteps; 

        % --- Update Step Counter ---
        nSteps = nSteps + 1;

        % --- Process Action ---
        % Assume agent Action is in range [-1, 1] (typical actor output)
        % Scale action to torque and clamp to physical limits defined in actInfo
        scaled_action = params.torqueScale * Action;
        torque = max(min(scaled_action, actInfo.UpperLimit), actInfo.LowerLimit);

        % --- Simulate one agent step (includes dynamics integration + frameskip) ---
        % Integrate dynamics for the step where torque is applied
        [omega, quat] = local_RK4_omega_quat(params.h, omega, quat, torque, params.I);

        % Free rotate for skipped frames (apply zero torque)
        for i = 1:params.frameskipNum
            [omega, quat] = local_RK4_omega_quat(params.h, omega, quat, [0; 0; 0], params.I);
        end

        % --- Prepare Next Observation ---
        NextObservation = [quat; omega]; % Format: [q1;q2;q3;q4; wx;wy;wz]

        % --- Calculate Reward ---
        qs_next = quat(4); % Scalar part of the new quaternion error
        omegaMag = norm(omega);

        % Update stabilization phase flag based on current error
        % Use abs(qs) as target can be reached with q4 near -1 too (180deg diff)
        if abs(qs_next) >= params.rewardThresholdQs
             isInStabilizationPhase = true;
        % else % Optional: Allow falling out of stabilization phase
        %      isInStabilizationPhase = false;
        end

        % Determine reward based on the phase
        if ~isInStabilizationPhase
            % Phase 1: Reward improvement in orientation error magnitude
            if abs(qs_next) > abs(qsPrev)
                 Reward = 0.1;
            else
                 Reward = -0.1;
            end
        else
            % Phase 2: Reward being close to target (Eq 15 logic)
            % Ensure element-wise power before dot product with alpha weights
            Reward = params.alpha' * (quat.^2); % -q1^2 -q2^2 -q3^2 + q4^2
        end

        % --- Check Termination/Truncation ---
        IsDoneVelocity = omegaMag > params.maxAngVelo; % Failure if velocity limit exceeded
        IsDoneSteps = nSteps >= params.maxEpisodeSteps;     % Truncation if max steps reached
        IsDone = IsDoneVelocity || IsDoneSteps;             % Episode ends for either reason

        % --- Apply Terminal Penalties/Rewards ---
        if IsDone
            if isempty(stepsBeyondDone) % First time IsDone is true this episode
                stepsBeyondDone = 0;
                if IsDoneVelocity
                    Reward = -25.0; % Velocity limit exceeded penalty
                elseif isInStabilizationPhase % Finished at max steps while stable
                    Reward = 10.0; % Success bonus
                % else % Finished at max steps but not stable (optional penalty?)
                     % Reward = -5.0; % Example penalty for truncation without success
                end
            else % If already done, subsequent steps (shouldn't happen in normal training) get 0 reward
                stepsBeyondDone = stepsBeyondDone + 1;
                Reward = 0.0;
            end
        else % If not done, reset the stepsBeyondDone counter
             stepsBeyondDone = [];
        end

        % --- Update LoggedSignals for next step ---
        LoggedSignals.quat = quat;
        LoggedSignals.omega = omega;
        LoggedSignals.nSteps = nSteps;
        LoggedSignals.qsPrev = qs_next; % Store current qs for next step's comparison
        LoggedSignals.stepsBeyondDone = stepsBeyondDone;
        LoggedSignals.isInStabilizationPhase = isInStabilizationPhase;
        LoggedSignals.MaxSteps = params.maxEpisodeSteps; 

    end % End nestedStepFunction

    % -------------------------------------------------------------------------

    function [InitialObservation, LoggedSignals] = nestedResetFunction()
        % Implements the reset logic for the environment

        % --- Reset State ---
        omega = zeros(3, 1); % Start at rest
        % Generate random initial error quaternion within specified angle range
        axisAngle = local_randomAxisAngle(params.minAngleSlew, params.maxAngleSlew);
        quat = local_axisAngleToQuat(axisAngle);

        InitialObservation = [quat; omega]; % Format: [q1;q2;q3;q4; wx;wy;wz]

        % --- Initialize LoggedSignals ---
        LoggedSignals.quat = quat;
        LoggedSignals.omega = omega;
        LoggedSignals.nSteps = int32(0);
        LoggedSignals.qsPrev = quat(4); % Initial scalar quaternion part
        LoggedSignals.stepsBeyondDone = []; % Initialize as empty
        LoggedSignals.isInStabilizationPhase = false; % Start in phase 1
        LoggedSignals.MaxSteps = params.maxEpisodeSteps; 
    end % End nestedResetFunction

    % =========================================================================
    % LOCAL HELPER FUNCTIONS (Callable by the nested functions above)
    % =========================================================================

    function axisAngle = local_randomAxisAngle(min_angle, max_angle)
        % Generates a random unit axis and angle within range [min_angle, max_angle] degrees
        phi_azimuth = 2*pi*rand();     % Azimuth angle [0, 2*pi]
        cos_theta = 2*rand() - 1;      % Cosine of inclination angle [-1, 1]
        theta = acos(cos_theta);       % Inclination angle [0, pi]

        % Convert spherical to Cartesian for unit axis vector
        x = sin(theta) * cos(phi_azimuth);
        y = sin(theta) * sin(phi_azimuth);
        z = cos(theta);
        axis_vec = [x; y; z];

        % Generate random angle in degrees and convert to radians
        angle_deg = min_angle + (max_angle - min_angle) * rand();
        angle_rad = deg2rad(angle_deg);

        axisAngle = [axis_vec; angle_rad];
    end
    % -------------------------------------------------------------------------
    function q = local_axisAngleToQuat(axisAngle)
        % Converts axis-angle representation to quaternion [q_vec; q_scalar]
        axis_vec = axisAngle(1:3);
        angle_rad = axisAngle(4);
        sinHalf = sin(angle_rad / 2);

        q_vec = axis_vec * sinHalf;
        q_scalar = cos(angle_rad / 2);

        q = [q_vec; q_scalar]; % Output as [q1; q2; q3; q4]
        q = local_normalizeQuat(q); % Ensure unit quaternion
    end
    % -------------------------------------------------------------------------
    function omegaDot = local_getOmegaDot(omega, torque, I_col_vec)
        % Calculates angular acceleration using Euler's equation
        % Assumes I_col_vec is the vector of principal moments of inertia
        Imat = diag(I_col_vec);        % Diagonal inertia matrix
        ImatInv = diag(1 ./ I_col_vec); % Inverse inertia matrix
        Iw = Imat * omega;
        % Euler's Equation: I * omega_dot + omega x (I * omega) = Torque
        % => omega_dot = I_inv * (Torque - omega x (I * omega))
        cross_term = cross(omega, Iw); % Calculate omega x (I*omega)
        omegaDot = ImatInv * (torque - cross_term);
    end
    % -------------------------------------------------------------------------
    function qDot = local_getQDot(omega, quat)
        % Calculates quaternion derivative using kinematic equation
        % Assumes quat = [q1; q2; q3; q4] (vector; scalar)
        % Kinematic Equation: q_dot = 0.5 * quatmultiply([omega; 0], quat)
        qDot = 0.5 * local_quatmultiply([omega; 0], quat);
    end
    % -------------------------------------------------------------------------
    function [omega1, q1] = local_RK4_omega_quat(h, omega0, q0, torque, I_col_vec)
        % 4th Order Runge-Kutta integration for coupled omega and quaternion dynamics

        % K1 calculation
        k1_omega_dot = local_getOmegaDot(omega0, torque, I_col_vec);
        k1_q_dot = local_getQDot(omega0, q0);

        % K2 calculation (intermediate step)
        omega_k2_in = omega0 + 0.5 * h * k1_omega_dot;
        q_k2_in_un = q0 + 0.5 * h * k1_q_dot;
        q_k2_in = local_normalizeQuat(q_k2_in_un); % Normalize intermediate
        k2_omega_dot = local_getOmegaDot(omega_k2_in, torque, I_col_vec);
        k2_q_dot = local_getQDot(omega_k2_in, q_k2_in);

        % K3 calculation (intermediate step)
        omega_k3_in = omega0 + 0.5 * h * k2_omega_dot;
        q_k3_in_un = q0 + 0.5 * h * k2_q_dot;
        q_k3_in = local_normalizeQuat(q_k3_in_un); % Normalize intermediate
        k3_omega_dot = local_getOmegaDot(omega_k3_in, torque, I_col_vec);
        k3_q_dot = local_getQDot(omega_k3_in, q_k3_in);

        % K4 calculation (intermediate step)
        omega_k4_in = omega0 + h * k3_omega_dot;
        q_k4_in_un = q0 + h * k3_q_dot;
        q_k4_in = local_normalizeQuat(q_k4_in_un); % Normalize intermediate
        k4_omega_dot = local_getOmegaDot(omega_k4_in, torque, I_col_vec);
        k4_q_dot = local_getQDot(omega_k4_in, q_k4_in);

        % Combine steps using RK4 formula
        omega1 = omega0 + (h / 6) * (k1_omega_dot + 2 * k2_omega_dot + 2 * k3_omega_dot + k4_omega_dot);
        q1_un = q0 + (h / 6) * (k1_q_dot + 2 * k2_q_dot + 2 * k3_q_dot + k4_q_dot);
        q1 = local_normalizeQuat(q1_un); % Final normalization
    end
    % -------------------------------------------------------------------------
    function qout = local_quatmultiply(q1, q2)
        % Quaternion Multiplication (Hamilton product)
        % Assumes q = [vector (3x1); scalar] format
        v1 = q1(1:3); s1 = q1(4);
        v2 = q2(1:3); s2 = q2(4);
        vout = s1*v2 + s2*v1 + cross(v1, v2);
        sout = s1*s2 - dot(v1, v2);
        qout = [vout; sout];
    end
    % -------------------------------------------------------------------------
    function q_norm = local_normalizeQuat(q)
        % Normalize Quaternion to ensure it remains a unit quaternion
        mag = norm(q);
        if mag < eps % Avoid division by near-zero
            q_norm = [0; 0; 0; 1]; % Return identity if magnitude is too small
        else
            q_norm = q / mag;
        end
    end

end % End main function createAttitudeControlEnv