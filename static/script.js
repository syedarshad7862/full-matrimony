function confirmDelete() {
    return confirm("Are you sure you want to delete your biodata? This action cannot be undone.")
}
console.log("this is working")
console.log("this is working")

function editProfile(id) {
    console.log("running inside edit fn")
    window.location.href = '/edit-profiles/' + id;
}
