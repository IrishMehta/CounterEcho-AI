document.addEventListener('DOMContentLoaded', () => {
    const usersList = document.getElementById('users-list');
    const tweetContent = document.getElementById('tweet-content');
    const postBtn = document.getElementById('post-btn');
    const feedStream = document.getElementById('feed-stream');

    let selectedUser = null;

    // 1. Fetch Users
    fetch('/users')
        .then(response => response.json())
        .then(users => {
            usersList.innerHTML = '';
            users.forEach((user, index) => {
                const item = createUserItem(user);
                usersList.appendChild(item);

                // Select first user by default
                if (index === 0) selectUser(user, item);
            });
        })
        .catch(err => console.error(err));

    function createUserItem(user) {
        const div = document.createElement('div');
        div.className = 'sidebar-user-item';
        div.onclick = () => selectUser(user, div);

        div.innerHTML = `
            <div class="user-avatar-small">
                <img src="${user.avatar}" alt="${user.name}">
            </div>
            <div class="user-details">
                <span class="user-name">${user.name}</span>
                <span class="user-handle">@${user.id}</span>
            </div>
        `;
        return div;
    }

    function selectUser(user, element) {
        selectedUser = user;
        document.querySelectorAll('.sidebar-user-item').forEach(el => el.classList.remove('selected'));
        element.classList.add('selected');

        // Update compose placeholder
        tweetContent.placeholder = `What is happening, ${user.name}?`;
        validateForm();
    }

    // 2. Input Validation
    tweetContent.addEventListener('input', validateForm);

    function validateForm() {
        if (selectedUser && tweetContent.value.trim().length > 0) {
            postBtn.disabled = false;
        } else {
            postBtn.disabled = true;
        }
    }

    // 3. Post Tweet
    postBtn.addEventListener('click', async () => {
        if (!selectedUser) return;

        const content = tweetContent.value;

        // Hide Compose Box
        document.querySelector('.compose-box').style.display = 'none';

        // Clear input
        tweetContent.value = '';
        postBtn.disabled = true;

        // Clear previous feed
        feedStream.innerHTML = '';

        // Render User's Tweet as MAIN POST
        const userTweetId = Date.now();
        renderMainTweet({
            id: userTweetId,
            name: selectedUser.name,
            handle: `@${selectedUser.id}`,
            avatar: selectedUser.avatar,
            text: content,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            date: new Date().toLocaleDateString([], { day: 'numeric', month: 'short', year: '2-digit' })
        });

        // Add "Most relevant replies" header
        const repliesHeader = document.createElement('div');
        repliesHeader.className = 'replies-header';
        repliesHeader.innerHTML = 'Most relevant replies <i class="fa-solid fa-chevron-down"></i>';
        feedStream.appendChild(repliesHeader);

        // Add loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-reply';
        loadingDiv.innerHTML = '<div class="loading-spinner"></div><span>CounterEcho AI is thinking...</span>';
        loadingDiv.style.cssText = 'display: flex; align-items: center; gap: 10px; padding: 20px; color: var(--text-secondary);';
        feedStream.appendChild(loadingDiv);

        // Call Backend
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tweet_type: 'NEW',
                    content: content,
                    camp: selectedUser.camp,
                    userId: selectedUser.id
                })
            });

            // Remove loading indicator
            loadingDiv.remove();

            // Check if response is ok
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', response.status, errorText);
                alert(`Server error (${response.status}): Please try again.`);
                return;
            }

            // Check for empty response
            const responseText = await response.text();
            if (!responseText) {
                console.error('Empty response from server');
                alert('Empty response from server. Please try again.');
                return;
            }

            let data;
            try {
                data = JSON.parse(responseText);
            } catch (parseErr) {
                console.error('Failed to parse response:', responseText);
                alert('Invalid response from server. Please try again.');
                return;
            }

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Render Counter-Echo Reply
            setTimeout(() => {
                renderReply({
                    id: userTweetId + 1,
                    name: 'CounterEcho AI',
                    handle: '@counter_echo',
                    avatar: 'https://api.dicebear.com/7.x/bottts/svg?seed=counterecho',
                    text: data.counterMessage,
                    replyingTo: `@${selectedUser.id}`
                });
            }, 800);

        } catch (err) {
            console.error('Error details:', err);
            // Remove loading indicator on error
            if (loadingDiv && loadingDiv.parentNode) {
                loadingDiv.remove();
            }
            alert('Failed to generate response: ' + err.message);
        }
    });

    function renderMainTweet(tweet) {
        const div = document.createElement('div');
        div.className = 'main-tweet';

        div.innerHTML = `
            <div class="main-tweet-header">
                <div class="main-tweet-user">
                    <div class="main-tweet-avatar">
                        <img src="${tweet.avatar}" alt="${tweet.name}">
                    </div>
                    <div class="main-tweet-names">
                        <span class="main-tweet-name">${tweet.name}</span>
                        <span class="main-tweet-handle">${tweet.handle}</span>
                    </div>
                </div>
                <i class="fa-solid fa-ellipsis" style="color: var(--text-secondary);"></i>
            </div>
            
            <div class="main-tweet-text">${tweet.text}</div>
            
            <div class="main-tweet-meta">
                ${tweet.time} · ${tweet.date} · <strong>87.6K</strong> Views
            </div>
            
            <div class="main-tweet-stats">
                <div class="stat-item"><strong>9</strong> Reposts</div>
                <div class="stat-item"><strong>1</strong> Quote</div>
                <div class="stat-item"><strong>1,461</strong> Likes</div>
            </div>
            
            <div class="main-tweet-actions">
                <div class="tweet-action"><i class="fa-regular fa-comment"></i></div>
                <div class="tweet-action"><i class="fa-solid fa-retweet"></i></div>
                <div class="tweet-action"><i class="fa-regular fa-heart"></i></div>
                <div class="tweet-action"><i class="fa-regular fa-bookmark"></i></div>
                <div class="tweet-action"><i class="fa-solid fa-share-nodes"></i></div>
            </div>
        `;

        feedStream.appendChild(div);
    }

    function renderReply(tweet) {
        const div = document.createElement('div');
        div.className = 'tweet is-reply';

        div.innerHTML = `
            <div class="tweet-avatar-container">
                <div class="tweet-avatar">
                    <img src="${tweet.avatar}" alt="${tweet.name}">
                </div>
                <!-- No thread line below the last reply -->
            </div>
            <div class="tweet-content">
                <div class="tweet-header">
                    <span class="tweet-name">${tweet.name}</span>
                    <span class="tweet-handle">${tweet.handle}</span>
                    <span class="tweet-handle">· 1m</span>
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.25rem;">Replying to <span style="color: var(--primary-color);">${tweet.replyingTo}</span></div>
                <div class="tweet-text">${tweet.text}</div>
                <div class="tweet-actions">
                    <div class="tweet-action"><i class="fa-regular fa-comment"></i> 0</div>
                    <div class="tweet-action"><i class="fa-solid fa-retweet"></i> 0</div>
                    <div class="tweet-action"><i class="fa-regular fa-heart"></i> 0</div>
                    <div class="tweet-action"><i class="fa-solid fa-chart-simple"></i> 0</div>
                    <div class="tweet-action"><i class="fa-solid fa-share-nodes"></i></div>
                </div>
            </div>
        `;
        feedStream.appendChild(div);

        // Scroll to bottom
        div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
});
