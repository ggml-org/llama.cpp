package com.example.llama.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.input.clearText
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SearchOff
import androidx.compose.material3.Button
import androidx.compose.material3.DockedSearchBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SearchBarDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.llama.ui.components.ModelCardFullExpandable
import com.example.llama.viewmodel.ModelScreenUiMode
import com.example.llama.viewmodel.ModelsViewModel
import com.example.llama.viewmodel.PreselectedModelToRun

@ExperimentalMaterial3Api
@Composable
fun ModelsSearchingScreen(
    preselection: PreselectedModelToRun?,
    viewModel: ModelsViewModel,
) {
    // Query states
    val textFieldState = viewModel.searchFieldState
    val searchQuery by remember(textFieldState) {
        derivedStateOf { textFieldState.text.toString() }
    }
    val queryResults by viewModel.queryResults.collectAsState()

    // Local UI states
    val focusRequester = remember { FocusRequester() }
    val keyboardController = LocalSoftwareKeyboardController.current
    val toggleSearchFocusAndIme: (Boolean) -> Unit = { show ->
        if (show) {
            focusRequester.requestFocus()
            keyboardController?.show()
        } else {
            focusRequester.freeFocus()
            keyboardController?.hide()
        }
    }

    // TODO-han.yin: remove after validation
//    LaunchedEffect (isSearchActive) {
//        if (isSearchActive) {
//            toggleSearchFocusAndIme(true)
//        }
//    }

    val handleExpanded: (Boolean) -> Unit = { expanded ->
        viewModel.toggleMode(
            if (expanded) ModelScreenUiMode.SEARCHING
            else ModelScreenUiMode.BROWSING
        )
        textFieldState.clearText()
    }

    Box(modifier = Modifier.fillMaxSize()) {
        DockedSearchBar(
            modifier = Modifier.align(Alignment.TopCenter),
            inputField = {
                SearchBarDefaults.InputField(
                    modifier = Modifier.focusRequester(focusRequester),
                    query = textFieldState.text.toString(),
                    onQueryChange = { textFieldState.edit { replace(0, length, it) } },
                    onSearch = {},
                    expanded = true,
                    onExpandedChange = handleExpanded,
                    leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                    trailingIcon = { Icon(Icons.Default.MoreVert, contentDescription = null) },
                    placeholder = { Text("Type to search your models") }
                )
            },
            expanded = true,
            onExpandedChange = handleExpanded
        ) {
            if (queryResults.isEmpty()) {
                if (searchQuery.isNotBlank()) {
                    // If no results under current query, show "no results" message
                    EmptySearchResultsView(
                        onClearSearch = {
                            textFieldState.clearText()
                            toggleSearchFocusAndIme(true)
                        }
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    contentPadding = PaddingValues(vertical = 12.dp, horizontal = 16.dp),
                ) {
                    items(items = queryResults, key = { it.id }) { model ->
                        ModelCardFullExpandable(
                            model = model,
                            isSelected = if (model == preselection?.modelInfo) true else null,
                            onSelected = { selected ->
                                if (selected) {
                                    toggleSearchFocusAndIme(false)
                                } else {
                                    viewModel.resetPreselection()
                                    toggleSearchFocusAndIme(true)
                                }
                            },
                            isExpanded = model == preselection?.modelInfo,
                            onExpanded = { expanded ->
                                viewModel.preselectModel(model, expanded)
                                toggleSearchFocusAndIme(!expanded)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptySearchResultsView(
    onClearSearch: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.SearchOff,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No matching models found",
            style = MaterialTheme.typography.headlineSmall
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Try a different search term",
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(onClick = onClearSearch) {
            Text("Clear Search")
        }
    }
}
