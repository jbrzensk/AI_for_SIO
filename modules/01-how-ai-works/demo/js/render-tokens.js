export function renderTokens(container, tokens, ids) {
  container.innerHTML = '';
  if (tokens.length === 0) {
    container.textContent = 'No tokens yet — type something and click Run.';
    return;
  }
  tokens.forEach((token, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';

    const label = document.createElement('span');
    label.className = 'token-chip-text';
    label.textContent = token;

    const idLabel = document.createElement('span');
    idLabel.className = 'token-chip-id';
    idLabel.textContent = String(ids[i]);

    chip.append(label, idLabel);
    container.appendChild(chip);
  });
}
