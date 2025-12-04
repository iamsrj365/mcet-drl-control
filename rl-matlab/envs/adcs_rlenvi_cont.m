% MATLAB version of the attitude control environment using RL Toolbox instead of OpenAI Gym

% Define observation and action specifications
obsInfo = rlNumericSpec([7 1]); % 4 quaternions + 3 angular velocity
obsInfo.Name = 'observations';
obsInfo.Description = 'Quaternion and angular velocity';

actInfo = rlNumericSpec([3 1], 'LowerLimit', -1, 'UpperLimit', 1); % Torque inputs
actInfo.Name = 'torque';

% Create environment
env = rlFunctionEnv(obsInfo, actInfo, @stepFunction, @resetFunction);

%% ---- Toolbox Functions ----

function axisAngle = randomAxisAngle(min_angle, max_angle)
    phi = 2*pi*rand();
    costheta = 2*rand() - 1;
    theta = acos(costheta);
    x = sin(theta) * cos(phi);
    y = sin(theta) * sin(phi);
    z = cos(theta);
    angle = deg2rad((max_angle - min_angle)*rand() + min_angle);
    axisAngle = [x; y; z; angle];
end

function q = axisAngleToQuat(axisAngle)
    x = axisAngle(1); y = axisAngle(2); z = axisAngle(3); ang = axisAngle(4);
    sinHalf = sin(ang/2);
    q = [x*sinHalf; y*sinHalf; z*sinHalf; cos(ang/2)];
end

function omegaDot = getOmegaDot(omega, torque, I)
    Imat = diag(I);
    ImatInv = diag(1 ./ I);
    Iw = Imat * omega;
    arg = torque + cross(Iw, omega);
    omegaDot = ImatInv * arg;
end

function qDot = getQDot(omega, quat)
    qv = quat(1:3);
    q4 = quat(4);
    omega_cross = [0, -omega(3), omega(2); omega(3), 0, -omega(1); -omega(2), omega(1), 0];
    qv_dot = q4 * omega + omega_cross * qv;
    q4_dot = -dot(omega, qv);
    qDot = [qv_dot; q4_dot];
end

function [omega1, q1] = RK4_omega_quat(h, omega0, q0, torque, I)
    kq = zeros(4,4);
    kw = zeros(3,4);

    kw(:,1) = h * getOmegaDot(omega0, torque, I);
    kq(:,1) = h * getQDot(omega0, q0);

    kw(:,2) = h * getOmegaDot(omega0 + 0.5*kw(:,1), torque, I);
    kq(:,2) = h * getQDot(omega0 + 0.5*kw(:,1), q0 + 0.5*kq(:,1));

    kw(:,3) = h * getOmegaDot(omega0 + 0.5*kw(:,2), torque, I);
    kq(:,3) = h * getQDot(omega0 + 0.5*kw(:,2), q0 + 0.5*kq(:,2));

    kw(:,4) = h * getOmegaDot(omega0 + kw(:,3), torque, I);
    kq(:,4) = h * getQDot(omega0 + kw(:,3), q0 + kq(:,3));

    omega1 = omega0 + (kw * [1;2;2;1]) / 6;
    q1 = q0 + (kq * [1;2;2;1]) / 6;
    q1 = q1 / norm(q1);
end

%% ---- Environment Callbacks ----

function [obs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals)
    I = [0.1, 0.1, 0.1]; % Moment of inertia
    h = 0.05; % Timestep

    omega = loggedSignals.omega;
    quat = loggedSignals.quat;
    
    [omega, quat] = RK4_omega_quat(h, omega, quat, action, I);
    
    obs = [quat; omega];

    % For demo, we reward minimizing angular velocity
    reward = -norm(omega);
    
    isDone = false; % could set condition based on angle to target
    
    loggedSignals.omega = omega;
    loggedSignals.quat = quat;
end

function [obs, loggedSignals] = resetFunction()
    axisAngle = randomAxisAngle(5, 15);
    quat = axisAngleToQuat(axisAngle);
    omega = 0.1 * (2*rand(3,1) - 1);

    obs = [quat; omega];

    loggedSignals.quat = quat;
    loggedSignals.omega = omega;
end


%--------------------------------------------------------------------------------------------------------------

% Actual environment starts


% Additional Parameters
% Mimic the init settings
params.torqueScale = 0.5;
params.frameskipNum = 20;
params.maxAngleSlew = 150;  % degrees
params.minAngleSlew = 30;   % degrees
params.I = [0.872, 0.115, 0.797];
params.h = 1/240;
params.qInitial = [0; 0; 0; 1];
params.alpha = [-1; -1; -1; 1];
params.maxEpisodeSteps = 500;
params.maxAngVelo = 0.5;

    % Global state variables for logging
    persistent qsPrev nSteps stepsBeyondDone qError qErrorDot omega

% Update stepFunction to reflect dynamics, reward and termination
function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals)
    global qsPrev nSteps stepsBeyondDone qError qErrorDot omega params

    nSteps = nSteps + 1;
    action = max(min(action, 1), -1);
    torque = params.torqueScale * action;

    [omega, qError] = RK4_omega_quat(@getOmegaDot, @getQDot, params.h, omega, qError, torque, params.I);

    % Free rotate for skipped frames
    for i = 1:params.frameskipNum
        [omega, qError] = RK4_omega_quat(@getOmegaDot, @getQDot, params.h, omega, qError, [0; 0; 0], params.I);
    end

    % No direct qErrorDot derivation in original, keeping zero for simplicity
    qErrorDot = zeros(4,1);

    % Updated state vector
    nextObs = [qError(4); qError(1:3); qErrorDot(4); qErrorDot(1:3); omega];

    % Compute angle and omega magnitude
    currAngle = 2 * acos(qError(4));
    omegaMag = norm(omega);
    isDone = omegaMag > params.maxAngVelo || nSteps >= params.maxEpisodeSteps;

    % Reward logic
    if ~isDone
        arg = currAngle / (0.14 * 2 * pi);
        if qError(4) > qsPrev
            reward = exp(-arg) - 0.5 * norm(action);
        else
            reward = exp(-arg) - 0.5 * norm(action) - 1;
        end
        if qError(4) >= 0.99999762
            reward = reward + 9;
        end
    elseif isempty(stepsBeyondDone)
        stepsBeyondDone = 0;
        if omegaMag > params.maxAngVelo
            reward = -25;
        elseif qError(4) >= 0.99999762
            reward = 50;
        else
            reward = 0;
        end
    else
        stepsBeyondDone = stepsBeyondDone + 1;
        reward = 0;
    end

    qsPrev = qError(4);
    loggedSignals.State = nextObs;
end

% Reset function
function [initialObs, loggedSignals] = resetFunction()
    global qsPrev nSteps stepsBeyondDone qError qErrorDot omega params

    nSteps = 0;
    stepsBeyondDone = [];

    goalAA = randomAxisAngle(params.minAngleSlew, params.maxAngleSlew);
    qError = axisAngleToQuat(goalAA);
    qErrorDot = zeros(4,1);
    omega = zeros(3,1);

    initialObs = [qError(4); qError(1:3); qErrorDot(4); qErrorDot(1:3); omega];
    qsPrev = qError(4);
    loggedSignals.State = initialObs;
end
